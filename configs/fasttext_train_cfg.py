import os
from datetime import datetime
from easydict import EasyDict

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))

cfg = EasyDict()

# Train params
cfg.batch_size = 128
cfg.lr = 0.5
cfg.lr_decay = None
cfg.momentum = 0.9
cfg.weight_decay = 0
cfg.max_epoch = 5
cfg.seed = 1
cfg.gpu = False

# Dictionary params
cfg.dict_params = EasyDict()
cfg.dict_params.MAX_VOCAB_SIZE = 30000000
cfg.dict_params.MAX_LINE_SIZE = 1024
cfg.dict_params.minCount = 5
cfg.dict_params.wordNgrams = 1
cfg.dict_params.bucket = 2000000
cfg.dict_params.minn = 3
cfg.dict_params.maxn = 6

# Model params
cfg.model = EasyDict()
cfg.model.dim = 100


cfg.save_freq = None
cfg.epoch_to_load = None
cfg.eval_before_training = True

cfg.experiment_name = f'fasttext_{cfg.batch_size}_{cfg.lr}_{cfg.dict_params.wordNgrams}-gram_' \
                      f'{datetime.now().strftime("%d_%m_%H%M%S")}'
cfg.log_dir = os.path.join(ROOT_DIR, 'data', 'fasttext', cfg.experiment_name)

cfg.plot_dir = os.path.join(cfg.log_dir, 'plots')
cfg.checkpoints_dir = os.path.join(cfg.log_dir, 'models')
cfg.pretrained_dir = cfg.checkpoints_dir