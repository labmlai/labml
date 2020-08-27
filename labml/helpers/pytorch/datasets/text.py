from pathlib import PurePath
from typing import List, Callable, Dict

import torch
from torch.utils.data import IterableDataset

from labml import monit


class TextDataset:
    itos: List[str]
    stoi: Dict[str, int]
    n_tokens: int
    train: str
    valid: str
    standard_tokens: List[str] = []

    @staticmethod
    def load(path: PurePath):
        with open(str(path), 'r') as f:
            return f.read()

    def __init__(self, path: PurePath, tokenizer: Callable, train: str, valid: str, test: str):
        self.test = test
        self.valid = valid
        self.train = train
        self.tokenizer = tokenizer
        self.path = path

        self.n_tokens = len(self.standard_tokens)
        self.stoi = {t: i for i, t in enumerate(self.standard_tokens)}

        with monit.section("Tokenize"):
            tokens = self.tokenizer(self.train)
            tokens = sorted(list(set(tokens)))

        for t in monit.iterate("Build vocabulary", tokens):
            self.stoi[t] = self.n_tokens
            self.n_tokens += 1

        self.itos = [''] * self.n_tokens
        for t, n in self.stoi.items():
            self.itos[n] = t

    def text_to_i(self, text: str) -> torch.Tensor:
        tokens = self.tokenizer(text)
        return torch.tensor([self.stoi[s] for s in tokens], dtype=torch.long)

    def __repr__(self):
        return f'{len(self.train)}, {len(self.valid)} - {str(self.path)}'


class SequentialDataLoader(IterableDataset):
    def __init__(self, *, text: str, dataset: TextDataset,
                 batch_size: int, seq_len: int):
        self.seq_len = seq_len
        data = dataset.text_to_i(text)
        n_batch = data.shape[0] // batch_size
        data = data.narrow(0, 0, n_batch * batch_size)
        data = data.view(batch_size, -1).t().contiguous()
        self.data = data

    def __len__(self):
        return self.data.shape[0] // self.seq_len

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx >= self.data.shape[0] - 1:
            raise StopIteration()

        seq_len = min(self.seq_len, self.data.shape[0] - 1 - self.idx)
        i = self.idx + seq_len
        data = self.data[self.idx: i]
        target = self.data[self.idx + 1: i + 1]
        self.idx = i
        return data, target
