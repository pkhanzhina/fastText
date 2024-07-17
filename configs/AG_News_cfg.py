import os
from easydict import EasyDict

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))

cfg = EasyDict()
cfg.main_dir = os.path.join(ROOT_DIR, 'data', 'AG News')

cfg.files = {
    'train': 'train.csv',
    'test': 'test.csv'
}

cfg.use_title = False
cfg.num_classes = 4
