import torch
import numpy as np

class SimpleSampler:
    def __init__(self, total, batch):
        self.total = total
        self.batch = batch
        self.curr = total
        self.ids = None

    def nextids(self):
        self.curr+=self.batch
        if self.curr + self.batch > self.total:
            self.ids = torch.LongTensor(np.random.permutation(self.total))
            self.curr = 0
        return self.ids[self.curr:self.curr+self.batch]


class ImageSampler:
    def __init__(self, total, batch=1):
        self.total = total
        self.batch = batch
        self.curr = 0
        self.ids = torch.LongTensor(torch.randperm(self.total))

    def nextids(self):
        img_ids = self.ids[self.curr:self.curr+self.batch]
        self.curr += self.batch
        if self.curr == self.total:
            self.curr = 0
            self.ids = torch.LongTensor(torch.randperm(self.total))
        return img_ids