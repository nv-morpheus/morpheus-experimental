# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
import pickle

import dgl
import torch
from model import HeteroRGCN
from sklearn.metrics import accuracy_score


def save_model(g, model, hyperparameters, model_dir):
    """Save trained model with graph & hyperparameters dict

    Args:
        g (_type_): dgl graph
        model (_type_): trained RGCN model
        model_dir (_type_): directory to save
        hyperparameters (_type_): hyperparameter for model training.
    """
    torch.save(model.state_dict(), os.path.join(model_dir, 'model.pt'))
    with open(os.path.join(model_dir, 'hyperparams.pkl'), 'wb') as f:
        pickle.dump(hyperparameters, f)
    with open(os.path.join(model_dir, 'graph.pkl'), 'wb') as f:
        pickle.dump(g, f)


def load_model(model_dir):
    """Load trained model, graph structure from given directory

    Args:
        model_dir (str path):directory path for trained model obj.

    Returns:
        _type_: model and graph structure.
    """

    with open(os.path.join(model_dir, "graph.pkl"), 'rb') as f:
        g = pickle.load(f)
    with open(os.path.join(model_dir, 'hyperparams.pkl'), 'rb') as f:
        hyperparameters = pickle.load(f)
    model = HeteroRGCN(g,
                       in_size=hyperparameters['in_size'],
                       hidden_size=hyperparameters['hidden_size'],
                       out_size=hyperparameters['out_size'],
                       n_layers=hyperparameters['n_layers'],
                       embedding_size=hyperparameters['embedding_size'],
                       target=hyperparameters['target_node'])
    model.load_state_dict(torch.load(os.path.join(model_dir, 'model.pt')))
    return model, g


def init_loaders(g_train, train_idx, test_idx, val_idx, g_test, target_node='authentication'):
    """Initialize dataloader and graph sampler. For training use neighbor sampling.

    Args:
        g_train (_type_): train graph
        train_idx (_type_): train feature index
        test_idx (_type_): test feature index
        val_idx (_type_): validation index
        g_test (_type_): test graph
        target_node (str, optional): target node. Defaults to 'authentication'.

    Returns:
        _type_: list of dataloaders
    """

    neighbor_sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
    full_sampler = dgl.dataloading.MultiLayerNeighborSampler([4, 3])

    train_dataloader = dgl.dataloading.NodeDataLoader(g_train, {target_node: train_idx},
                                                      neighbor_sampler,
                                                      batch_size=1000,
                                                      shuffle=False,
                                                      drop_last=False,
                                                      num_workers=0)

    test_dataloader = dgl.dataloading.NodeDataLoader(g_test, {target_node: test_idx},
                                                     full_sampler,
                                                     batch_size=100,
                                                     shuffle=False,
                                                     drop_last=False,
                                                     num_workers=0)

    val_dataloader = dgl.dataloading.NodeDataLoader(g_train, {target_node: val_idx},
                                                    neighbor_sampler,
                                                    batch_size=1000,
                                                    shuffle=False,
                                                    drop_last=False,
                                                    num_workers=0)

    return train_dataloader, val_dataloader, test_dataloader


def train(model,
          loss_func,
          train_dataloader,
          labels,
          optimizer,
          feature_tensors,
          target_node='authentication',
          device='cpu'):
    """Training GNN model

    Args:
        model : RGCN model
        loss_func : loss function
        train_dataloader : train dataloader class
        labels: training label
        optimizer : optimizer for training
        feature_tensors : node features
        target_node (str, optional): target node embedding. Defaults to 'authentication'.
        device (str, optional): _description_. Defaults to 'cpu'.

    Returns:
        _type_: training accuracy and training loss
    """
    model.train()
    train_loss = 0.0
    for i, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
        seed = output_nodes[target_node]
        blocks = [b.to(device) for b in blocks]
        nid = blocks[0].srcnodes[target_node].data[dgl.NID]
        input_features = feature_tensors[nid].to(device)

        logits = model(blocks, input_features)
        loss = loss_func(logits, labels[seed])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_acc = accuracy_score(logits.argmax(1), labels[seed].long()).item()
    return train_acc, train_loss


@torch.no_grad()
def evaluate(model, eval_loader, feature_tensors, target_node, device='cpu'):
    """Takes trained RGCN model and input dataloader & produce logits and embedding.

    Args:
        model: trained HeteroRGCN model object
        eval_loader : evaluation dataloader
        feature_tensors : test feature tensor
        target_node (_type_): target node encoding.
        device (str, optional): device runtime. Defaults to 'cpu'.

    Returns:
        _type_: logits, index & output embedding.
    """
    model.eval()
    eval_logits = []
    eval_seeds = []
    embedding = []

    for input_nodes, output_nodes, blocks in eval_loader:

        seed = output_nodes[target_node]

        nid = blocks[0].srcnodes[target_node].data[dgl.NID]
        blocks = [b.to(device) for b in blocks]
        input_features = feature_tensors[nid].to(device)
        logits, embedd = model.infer(blocks, input_features)
        eval_logits.append(logits.cpu().detach())
        eval_seeds.append(seed)
        embedding.append(embedd)

    eval_logits = torch.cat(eval_logits)
    eval_seeds = torch.cat(eval_seeds)
    embedding = torch.cat(embedding)
    return eval_logits, eval_seeds, embedding
