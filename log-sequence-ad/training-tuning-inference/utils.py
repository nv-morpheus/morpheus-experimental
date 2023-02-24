# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


def split(dataset, ratio):
    # split dataset into two based on given ratio.
    total = len(dataset)
    test = int(total * ratio)
    return dataset[:test], dataset[test:]


class LogDataset(Dataset):
    """Create torch Dataset

    Parameters
    ----------
    Dataset : torch.util.Dataset
        _description_
    """

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


def get_iter(X, y, batch_size=32, shuffle=True):
    """Create dataloader iterator

    Parameters
    ----------
    X : torch.tensor
        training sample
    y : torch.tensor
        ground truth label
    batch_size : int, optional
        batch size, by default 32
    shuffle : bool, optional
        _description_, by default True

    Returns
    -------
    DataLoader
        iterator for dataset dataloader
    """
    dataset = LogDataset(X, y)
    if shuffle:
        iter = DataLoader(dataset, batch_size, shuffle=True, worker_init_fn=np.random.seed(42))
    else:
        iter = DataLoader(dataset, batch_size)
    return iter
