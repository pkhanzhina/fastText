import os
from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
    def __init__(self, log_dir):
        self.logger = SummaryWriter(log_dir=log_dir)

    def log_metrics(self, names, metrics, step):
        for name, metric in zip(names, metrics):
            self.logger.add_scalar(name, metric, global_step=step)

    def log_configs(self, names, cfgs):
        pass