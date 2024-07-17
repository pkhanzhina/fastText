import os
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import json
import torch.multiprocessing as mp
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dataset.AG_News import AG_News
from models.fasttext import FastText
from utils.tensorboard_logger import TensorBoardLogger
from utils.accuracy import accuracy, plot_confusion_matrix


class MultithreadedTrainer:
    def __init__(self, cfg):
        self.cfg = cfg

        self.start_epoch, self.max_epoch = 0, cfg.max_epoch

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
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.cfg.lr, momentum=self.cfg.momentum, weight_decay=self.cfg.weight_decay)

        self.model.to(self.device)
        self.model.train()
        print('number of trainable params:\t', sum(p.numel() for p in self.model.parameters() if p.requires_grad),
              f'\tdevice:\t{self.device}')

    def __prepare_data(self):
        self.train_dataset = AG_News(self.cfg.dataset, 'train', dict_params=self.cfg.dict_params)
        self.test_dataset = AG_News(self.cfg.dataset, 'test', dict_=self.train_dataset.dict)

        # self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.cfg.batch_size, shuffle=True, drop_last=True)
        # self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.cfg.batch_size, shuffle=False, drop_last=False)

    def console_log(self, epoch, names, metrics):
        print(f"[{epoch}]:\t", ',\t'.join(['{!s}: {:.4f}'.format(n, m) for n, m in zip(names, metrics)]))

    def multithread_fit(self):
        mp.set_start_method('spawn', force=True)
        self.model.share_memory()
        self.cfg.start_epoch = self.start_epoch
        processes = []
        for rank in range(self.cfg.threads):
            loader = DataLoader(
                self.train_dataset,
                sampler=DistributedSampler(
                    dataset=self.train_dataset,
                    num_replicas=self.cfg.threads,
                    rank=rank),
                batch_size=self.cfg.batch_size,
                drop_last=True,
                pin_memory=True if self.cfg.gpu else False
            )

            p = mp.Process(target=self.train, args=(loader, self.model, self.device, self.cfg))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

    @staticmethod
    def train(loader, model, device, args):
        criterion = nn.NLLLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        for epoch in range(args.start_epoch, args.max_epoch):
            MultithreadedTrainer.train_epoch(loader, model, device, optimizer, criterion, epoch)

    @staticmethod
    def train_epoch(loader, model, device, optimizer, criterion, epoch):
        pid = os.getpid()
        for batch_idx, batch in enumerate(loader):
            optimizer.zero_grad()
            output = model(batch['data'].to(device))
            loss = criterion(output, batch['label'].to(device))
            loss.backward()
            optimizer.step()

            output, labels = output.detach().cpu().numpy(), batch['label'].numpy()
            acc = accuracy(output, labels)
            if batch_idx % 100 == 0:
                print('{} [{}: {}/{}]\t loss: {:.4f}, acc: {:.4f}'.format(pid, epoch, batch_idx, len(loader), loss.item(), acc))


if __name__ == '__main__':
    from configs.fasttext_train_cfg import cfg as train_cfg
    from configs.AG_News_cfg import cfg as dataset_cfg
    import time

    train_cfg.dataset = dataset_cfg
    train_cfg.batch_size = 128
    train_cfg.threads = 2
    train_cfg.max_epoch = 1
    train_cfg.momentum = 0

    trainer = MultithreadedTrainer(train_cfg)
    s = time.time()
    trainer.multithread_fit()
    f = time.time()
    print(train_cfg.batch_size, f - s)
