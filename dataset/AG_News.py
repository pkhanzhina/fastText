import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from dataset.dictionary import Dictionary
from dataset.preprocessing import clean_str, tokenization


class AG_News(Dataset):
    def __init__(self, cfg, data_type, dict_params=None, dict_=None, remove_stopwords=True):
        self.remove_stopwords = remove_stopwords

        path = os.path.join(cfg.main_dir, cfg.files[data_type])
        df = pd.read_csv(path)

        self.texts = df['Title'] + ' ' + df['Description'] if cfg.use_title else df['Description']
        self.labels = df['Class Index'] - 1

        tqdm.pandas(desc='Data preprocessing')
        self.texts = self.texts.progress_apply(self.preprocessing)

        self.dict = dict_
        if dict_ is None:
            self.dict = Dictionary(dict_params)
            self.dict.readFromFile(self.texts)

    def __getitem__(self, idx):
        txt = self.texts[idx]
        words = torch.LongTensor(self.dict.getLine(txt))
        return {
            'data': words,
            'label': self.labels[idx]
        }

    def __len__(self):
        return len(self.labels)

    def preprocessing(self, s):
        s = clean_str(s)
        tokens = tokenization(s, self.remove_stopwords)
        return tokens


if __name__ == '__main__':
    from configs.AG_News_cfg import cfg

    dataset = AG_News(cfg, 'train')

    print()
