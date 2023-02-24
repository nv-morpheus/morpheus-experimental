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
import torch.nn as nn
import utils


class LogLSTM(nn.Module):
    """LSTM model for sequence binary classification

    Parameters
    ----------
    nn : torch.nn

    """

    def __init__(self, matrix_embeddings, vocab_dim, output_dim, emb_dim, hid_dim, n_layers, dropout=0.05):
        """

        Parameters
        ----------
        matrix_embeddings : torch.tensor
            word2vec embedding
        vocab_dim : int
            vocab dimension
        output_dim : int
            output dim
        emb_dim : int
            embedding dim
        hid_dim : int
            hidden layer dim
        n_layers : int
            number of layers
        dropout : float, optional
            dropout size, by default 0.05
        """
        super().__init__()

        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.vocab_dim = vocab_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding.from_pretrained(matrix_embeddings)
        self.embedding.requires_grad = False
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, bidirectional=True, batch_first=True)
        self.fc_out = nn.Linear(hid_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        """Forward pass

        Parameters
        ----------
        input : nn.torch
            log input as tensor

        Returns
        -------
        _type_
            binary prediction
        """
        embedded = self.dropout(self.embedding(input))
        output, (_, _) = self.rnn(embedded)
        prediction = self.fc_out(output[:, -1, :])
        return prediction


def count_parameters(model):
    # count model parameters
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(model, iterator, optimizer, criterion, device):
    """Train LogLSTM model

    Parameters
    ----------
    model : LogLSTM
    iterator : DataLoader
        training input
    optimizer : optimizer
        training optimizer
    criterion : torch.nn
        loss criteria
    device : str
        host device

    Returns
    -------
    _type_
        _description_
    """

    model.train()

    epoch_loss = 0
    for (i, batch) in enumerate(iterator):
        src = batch[0].to(device)
        trg = batch[1].to(device)
        optimizer.zero_grad()
        output = model(src)

        trg = trg.view(-1)
        loss = criterion(output, trg.to(dtype=torch.long))
        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion, device):
    """Evaluate model

    Parameters
    ----------
    model : LogLSTM
       trained model
    iterator : DataLoader
    criterion : torch.nn
        loss criteria
    device : str
        host

    Returns
    -------
    _type_
       epoch loss
    """

    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for (i, batch) in enumerate(iterator):
            src = batch[0].to(device)
            trg = batch[1].to(device)
            output = model(src)
            trg = trg.view(-1)
            loss = criterion(output, trg.to(dtype=torch.long))

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def test(model, iterator, device):
    """_summary_

    Parameters
    ----------
    model : _type_
        _description_
    iterator : _type_
        _description_
    device : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    model.eval()
    y = []
    y_pre = []
    with torch.no_grad():
        for (i, batch) in enumerate(iterator):
            src = batch[0].to(device)
            trg = batch[1].to(device)
            output = model(src)
            result = list(torch.argmax(output, dim=1).detach().cpu().numpy())
            y_pre.extend(result)
            y.extend(list(trg.detach().cpu().numpy()))

    return y, y_pre


def epoch_time(start_time, end_time):
    # utility to create elapsed time of an epoch
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def model_inference(model, device, test_data):
    # model inference for test_data
    model.eval()
    dummy_label = [0 for _ in range(len(test_data))]
    X_all = torch.tensor(test_data, requires_grad=False).long()
    y_all = torch.tensor(dummy_label).reshape(-1, 1).long()
    return test(model, utils.get_iter(X_all, y_all, shuffle=False), device)


def model_precision(model, device, lst_n, lst_ab):
    """Create dataloader of test data and return inference
       result based on trained model.

    Parameters
    ----------
    model : LogLSTM model
        trained model
    device : str
        host device
    lst_n : list
        normal test case list
    lst_ab : list
        abnormal test case list

    Returns
    -------
    list
        prediction result

    """
    model.eval()
    y_all = []
    X_all = []
    y_n = [0 for _ in range(len(lst_n))]
    y_ab = [1 for _ in range(len(lst_ab))]
    y_all.extend(y_ab)
    y_all.extend(y_n)
    X_all.extend(lst_ab)
    X_all.extend(lst_n)
    X_all = torch.tensor(X_all, requires_grad=False).long()
    y_all = torch.tensor(y_all).reshape(-1, 1).long()
    all_iter = utils.get_iter(X_all, y_all, shuffle=False)

    return test(model, all_iter, device)


def ratio_abnormal_sequence(df, window_size=100, ratio=0.1):
    """Return abnormal sequence

    Parameters
    ----------
    df : pd.DataFrame
        input dataframe
    window_size : int, optional
        _description_, by default 100
    ratio : float, optional
        _description_, by default 0.1

    Returns
    -------
    _type_
        abnormal proportion
    """
    lst_sum = list(np.hstack(df['Key_label'].values).reshape(-1, window_size).sum(axis=1))
    df['Abnormal_Ratio'] = lst_sum
    return df.loc[df['Abnormal_Ratio'] <= ratio * window_size]
