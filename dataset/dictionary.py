import numpy as np
from collections import Counter
from tqdm.auto import tqdm

from utils.constants import *


def gethash(str_):
    h = np.uint32(hval)
    for c in str_:
        h = h ^ np.uint32(np.int8(ord(c)))
        h = np.uint32(h * FNV_32_PRIME)
    return h


class entry:
    def __init__(self, w):
        self.word = w
        self.count = 1
        self.subwords = []


class Dictionary:

    def __init__(self, cfg):

        self.cfg = cfg
        self.MAX_VOCAB_SIZE = cfg.MAX_VOCAB_SIZE
        self.MAX_LINE_SIZE = cfg.MAX_LINE_SIZE
        self.bucket = cfg.bucket
        self.wordNgrams = cfg.wordNgrams
        self.minn, self.maxn = cfg.minn, cfg.maxn
        self.minCount = cfg.minCount
        self.padding_idx = -1

        self.words_ = []
        self.word2int_ = np.full(self.MAX_VOCAB_SIZE, -1)
        self.size_, self.ntokens_ = 0, 0

        self.initNgrams()

    def nwords(self):
        return self.size_

    def getId(self, w, h):
        _id = self._find(w, h)
        return self.word2int_[_id]

    def getSubwords(self, i):
        return self.words_[i].subwords

    def find(self, w):
        return self._find(w, gethash(w))

    def _find(self, w, h):
        _id = h % self.MAX_VOCAB_SIZE
        while self.word2int_[_id] != -1 and self.words_[self.word2int_[_id]].word != w:
            _id = (_id + 1) % self.MAX_VOCAB_SIZE
        return _id

    def add(self, w):
        h = self.find(w)
        self.ntokens_ += 1
        if self.word2int_[h] == -1:
            self.words_.append(entry(w))
            self.word2int_[h] = self.size_
            self.size_ += 1
        else:
            self.words_[self.word2int_[h]].count += 1

    def threshold(self, t):
        sorted(self.words_, key=lambda x: -x.count)
        self.words_ = [w for w in self.words_ if w.count >= t]

        self.size_ = 0
        self.word2int_ = np.full(self.MAX_VOCAB_SIZE, -1)
        for it in self.words_:
            h = self.find(it.word)
            self.word2int_[h] = self.size_
            self.size_ += 1

    def initNgrams(self):
        for i in range(self.size_):
            self.words_[i].subwords = [i]
            self.computeSubwords(BOW + self.words_[i].word + EOW, self.words_[i].subwords)

    def computeSubwords(self, word, ngrams):
        for i in range(len(word)):
            # if ord(word[i]) & 0xC0 == 0x80:  # TODO: зачем??
            #     continue
            ngram = ''
            j, n = i, 1
            while j < len(word) and n <= self.maxn:
                ngram += word[j]
                j += 1
                # while j < len(word) and (ord(word[j]) & 0xC0 == 0x80):  # TODO: зачем??
                #     ngram += word[j]
                #     j += 1
                if n >= self.minn and not (n == 1 and (i == 0 or j == len(word))):
                    h = gethash(ngram) % self.bucket
                    self.pushHash(ngrams, h)
                n += 1

    def pushHash(self, hashes, _id):
        hashes.append(self.size_ + _id)

    def addWordNgrams(self, line, hashes, n):
        for i in range(len(hashes)):
            h = hashes[i]
            for j in range(i + 1, min(len(hashes), i + n)):
                h = np.uint64(h * 116049371 + hashes[j])  # TODO: to constants
                self.pushHash(line,  np.uint64(h % self.bucket))

    def addSubwords(self, line, token, wid):
        if wid < 0:
            self.computeSubwords(BOW + token + EOW, line)
        else:
            if self.maxn <= 0:
                line.append(wid)
            else:
                ngrams = self.getSubwords(wid)
                line.extend(ngrams)

    def readFromFile(self, corpus):
        minThreshold = 1
        pbar = tqdm(corpus, desc='Prepare dictionary')
        for txt in pbar:
            for word in txt:
                self.add(word)
                if self.size_ > 0.75 * self.MAX_VOCAB_SIZE:
                    minThreshold += 1
                    self.threshold(minThreshold)

        self.threshold(self.minCount)
        self.initNgrams()

        self.padding_idx = self.size_ + self.bucket

    def getLine(self, txt):
        word_hashes, words = [], []
        for token in txt:
            h = self.find(token)
            wid = self.getId(token, h)
            if wid >= 0:
                self.addSubwords(words, token, wid)
                word_hashes.append(h)
        self.addWordNgrams(words, word_hashes, self.wordNgrams)
        if len(words) < self.MAX_LINE_SIZE:
            words = words + [self.padding_idx] * (self.MAX_LINE_SIZE - len(words))
        elif len(words) > self.MAX_LINE_SIZE:
            words = words[:self.MAX_LINE_SIZE]
        return words


if __name__ == '__main__':
    from dataset.preprocessing import preprocessing
    import pandas as pd
    from configs.fasttext_train_cfg import cfg as train_cfg

    df = pd.DataFrame(data={
        'labels': [0, 0],
        # 'text': ['cat']
        'text': ['the man went out for a walk'.split(), 'the children sat around the fire'.split()]
    })
    texts = df['text']
    # texts = df['text'].apply(preprocessing)

    print()

    dictionary = Dictionary(train_cfg.dict_params)
    dictionary.readFromFile(texts)

    words = dictionary.getLine('the man went out for a walk'.split())

    print()
