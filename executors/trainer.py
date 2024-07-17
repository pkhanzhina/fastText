import json
import os
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm

from dataset.AG_News import AG_News
from models.fasttext import FastText
from utils.accuracy import accuracy, plot_confusion_matrix
from utils.tensorboard_logger import TensorBoardLogger


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg

        self.start_epoch, self.global_step = 0, 0
        self.max_epoch = cfg.max_epoch

        self.__prepare_data()
        self.__prepare_model()

        self.logger = TensorBoardLogger(os.path.join(self.cfg.log_dir, 'logs'))
        if not os.path.exists(self.cfg.plot_dir):
            os.makedirs(self.cfg.plot_dir)
        with open(os.path.join(cfg.log_dir, 'config.json'), 'w') as fp:
            json.dump(self.cfg, fp)

    def __prepare_model(self):
        self.device = torch.device('cuda' if self.cfg.gpu else 'cpu')
        self.model = FastText(
            input_size=self.train_dataset.dict.nwords() + self.cfg.dict_params.bucket,
            dim=self.cfg.model.dim,
            count_classes=self.cfg.dataset.num_classes,
            padding_idx=self.train_dataset.dict.padding_idx
        )
        self.criterion = nn.NLLLoss()
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.cfg.lr, momentum=self.cfg.momentum, weight_decay=self.cfg.weight_decay)
        if self.cfg.lr_decay is not None:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=2, gamma=self.cfg.lr_decay)

        self.model.to(self.device)
        print('number of trainable params:\t', sum(p.numel() for p in self.model.parameters() if p.requires_grad),
              f'\tdevice:\t{self.device}')

        if self.cfg.epoch_to_load is not None:
            self.start_epoch = self._load_model(self.cfg.epoch_to_load)

    def __prepare_data(self):

        self.train_dataset = AG_News(self.cfg.dataset, 'train', dict_params=self.cfg.dict_params)
        self.test_dataset = AG_News(self.cfg.dataset, 'test', dict_=self.train_dataset.dict)

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.cfg.batch_size, shuffle=True, drop_last=True)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.cfg.batch_size, shuffle=False)

    def console_log(self, epoch, names, metrics):
        print(f"[{epoch}] ", ',\t'.join(['{!s}: {:.4f}'.format(n, m) for n, m in zip(names, metrics)]))

    def _dump_model(self, epoch):
        state_dict = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'scheduler': self.scheduler
        }
        if not os.path.exists(self.cfg.checkpoints_dir):
            os.makedirs(self.cfg.checkpoints_dir)
        path_to_save = os.path.join(self.cfg.checkpoints_dir, f'epoch-{epoch}.pt')
        torch.save(state_dict, path_to_save)

    def _load_model(self, epoch):
        path = os.path.join(self.cfg.pretrained_dir, f"epoch-{epoch}.pt")
        start_epoch = 0
        try:
            state_dict = torch.load(path)
            self.model.load_state_dict(state_dict['model_state_dict'])
            self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
            self.scheduler.load_state_dict(state_dict['scheduler'])
            self.global_step = state_dict['global_step']
            start_epoch = state_dict['epoch']
        except Exception as e:
            print(e)
        return start_epoch

    def make_training_step(self, data, labels, update=True):
        pred = self.model(data.to(self.device))
        loss = self.criterion(pred, labels.to(self.device))
        if update:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        return loss.item(), pred.detach().cpu()

    def train_epoch(self, epoch):
        self.model.train()
        pbar = tqdm(self.train_dataloader)
        for batch in pbar:
            loss, output = self.make_training_step(batch['data'], batch['label'])

            output, labels = output.numpy(), batch['label'].numpy()
            acc = accuracy(output, labels)

            metrics = [loss, self.optimizer.param_groups[0]['lr'], acc]
            names = ['train/loss', 'train/lr', 'train/acc']

            self.logger.log_metrics(names=names, metrics=metrics, step=self.global_step)
            pbar.set_description('[{}/{}] loss: {:.4f}, acc: {:.4f}'.format(epoch, self.max_epoch, loss, acc))

            self.global_step += 1

    def fit(self):
        if self.cfg.eval_before_training:
            self.evaluate('test', step=self.start_epoch)
            self.evaluate('train', step=self.start_epoch)
        for epoch in range(self.start_epoch, self.max_epoch):
            self.train_epoch(epoch)
            self.evaluate('test', step=epoch + 1)
            self.evaluate('train', step=epoch + 1)
            if self.cfg.lr_decay is not None:
                self.scheduler.step()
            if self.cfg.save_freq is not None and epoch % self.cfg.save_freq == 0:
                self._dump_model(epoch + 1)
        if self.cfg.save_freq is not None:
            self._dump_model(self.max_epoch)

    def overfit(self):
        self.model.train()
        pbar = tqdm(range(100))
        batch = next(iter(self.train_dataloader))
        for _iter in pbar:
            loss, output = self.make_training_step(batch['data'], batch['label'])
            output, labels = output.numpy(), batch['label'].numpy()
            acc = accuracy(output, labels)
            pbar.set_description('loss: {:.4f}, acc: {:.4f}'.format(loss, acc))

    @torch.no_grad()
    def evaluate(self, data_type='test', step=None):
        step = self.global_step if step is None else step
        loader = self.test_dataloader if data_type == 'test' else self.train_dataloader

        losses, outputs, labels = [], [], []
        nrof_samples = 0

        self.model.eval()

        for i, batch in enumerate(tqdm(loader, desc=f'evaluation on {data_type} set')):
            loss, output = self.make_training_step(batch['data'], batch['label'], update=False)

            nrof_samples += output.size(0)
            losses.append(loss * output.size(0))
            outputs.append(output)
            labels.append(batch['label'])

        loss = sum(losses) / nrof_samples

        outputs, labels = torch.vstack(outputs).numpy(), torch.hstack(labels).numpy()
        acc = accuracy(outputs, labels)
        plot_confusion_matrix(outputs, labels, title=f"{data_type} (epoch {step})",
                              path_to_save=os.path.join(self.cfg.plot_dir, f"{data_type}_{step}.png"))

        self.console_log(step, names=['loss', 'acc'], metrics=[loss, acc])
        self.logger.log_metrics(names=[f'eval_{data_type}/loss', f'eval_{data_type}/acc'], metrics=[loss, acc], step=step)


if __name__ == '__main__':
    from configs.fasttext_train_cfg import cfg as train_cfg
    from configs.AG_News_cfg import cfg as dataset_cfg

    train_cfg.dataset = dataset_cfg

    trainer = Trainer(train_cfg)
    trainer.fit()
