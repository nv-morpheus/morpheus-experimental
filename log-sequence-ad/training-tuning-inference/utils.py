import pandas as pd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

def split(dataset, ratio):
    total = len(dataset)
    test = int(total*ratio)
    return dataset[:test], dataset[test:]

class LogDataset(Dataset):
    def __init__(self, train_features, train_labels):
        super().__init__()
        self.train_features = train_features
        self.train_labels = train_labels

    def __len__(self):
        return len(self.train_features)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return (self.train_features[idx], self.train_labels[idx])

def get_iter(X, y, batch_size = 32, shuffle = True):
    dataset = LogDataset(X,y)
    if shuffle == True:
        iter = DataLoader(dataset, batch_size, shuffle = True, worker_init_fn=np.random.seed(42))
    else:
        iter = DataLoader(dataset, batch_size)
    return iter

def get_iter_hdfs(X, y, w2v_dic, train_dict, batch_size = 32, shuffle = True):
    dataset = LogDataset(X,y)
    PADDING_VALUE=w2v_dic[train_dict['pad']]

    def pad_collate(batch):
        (X, y) = zip(*batch)
        X_lens = [len(x) for x in X]
        X = [torch.LongTensor(i) for i in X]
        X_pad = pad_sequence(X, batch_first=True, padding_value=PADDING_VALUE)
        y = torch.Tensor(y)
        return X_pad, X_lens, y

    if shuffle == True:
        iter = DataLoader(dataset, batch_size, shuffle = True, worker_init_fn=np.random.seed(42), collate_fn=pad_collate)
    else:
        iter = DataLoader(dataset, batch_size, collate_fn=pad_collate)
    return iter