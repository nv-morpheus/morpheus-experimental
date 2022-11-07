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

import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F


class HeteroRGCNLayer(nn.Module):

    def __init__(self, in_size, out_size, etypes):
        super(HeteroRGCNLayer, self).__init__()
        # W_r for each relation
        input_sizes = [in_size] * len(etypes) if type(in_size) == int else in_size
        self.weight = nn.ModuleDict({name: nn.Linear(in_dim, out_size) for name, in_dim in zip(etypes, input_sizes)})

    def forward(self, G, feat_dict):
        # The input is a dictionary of node features for each type
        funcs = {}
        for srctype, etype, dsttype in G.canonical_etypes:
            # Compute W_r * h
            if srctype in feat_dict:
                Wh = self.weight[etype](feat_dict[srctype])
                # Save it in graph for message passing
                G.nodes[srctype].data['Wh_%s' % etype] = Wh
                funcs[etype] = (fn.copy_u('Wh_%s' % etype, 'm'), fn.mean('m', 'h'))

        G.multi_update_all(funcs, 'sum')
        # return the updated node feature dictionary
        return {ntype: G.dstnodes[ntype].data['h'] for ntype in G.ntypes if 'h' in G.dstnodes[ntype].data}


class HeteroRGCN(nn.Module):

    def __init__(self, g, in_size, hidden_size, out_size, n_layers, embedding_size, device='cpu', target='transaction'):
        super(HeteroRGCN, self).__init__()
        self.target = target
        # Use trainable node embeddings as featureless inputs.
        embed_dict = {
            ntype: nn.Parameter(torch.Tensor(g.number_of_nodes(ntype), embedding_size))
            for ntype in g.ntypes if ntype != self.target
        }
        for key, embed in embed_dict.items():
            nn.init.xavier_uniform_(embed)
        self.embed_dict = {ntype: embedding.to(device) for ntype, embedding in embed_dict.items()}
        # create layers

        in_sizes = [in_size if src_type == self.target else embedding_size for src_type, _, _ in g.canonical_etypes]
        layers = [HeteroRGCNLayer(in_sizes, hidden_size, g.etypes)]
        # hidden layers
        for i in range(n_layers - 1):
            layers.append(HeteroRGCNLayer(hidden_size, hidden_size, g.etypes))

        # output layer
        layers.append(nn.Linear(hidden_size, out_size))
        self.layers = nn.Sequential(*layers)
        self.device = device
        self.g_embed = None

    def embed(self, g, features):
        # get embeddings for all node types. for user node type, use passed in user features
        h_dict = {self.target: features}
        for ntype in self.embed_dict:
            if g[0].number_of_nodes(ntype) > 0:
                h_dict[ntype] = self.embed_dict[ntype][g[0].nodes(ntype).to(self.device)]

        # pass through all layers
        for i, layer in enumerate(self.layers[:-1]):
            if i != 0:
                h_dict = {k: F.leaky_relu(h) for k, h in h_dict.items()}
            h_dict = layer(g[i], h_dict)
        self.g_embed = h_dict
        return h_dict[self.target]

    def forward(self, g, features):
        """ Input is Graph g and features for target node.
        """
        return self.layers[-1](self.embed(g, features))

    def infer(self, g, features):
        embedding = self.embed(g, features)
        predictions = self.layers[-1](embedding)
        return nn.Sigmoid()(predictions), embedding


def save_model(g, model, model_dir):
    torch.save({'model_state_dict': model.state_dict()}, os.path.join(model_dir, 'model.pt'))
    # with open(os.path.join(model_dir, 'model_hyperparams.pkl'), 'wb') as f:
    #     pickle.dump(hyperparams, f)
    with open(os.path.join(model_dir, 'graph.pkl'), 'wb') as f:
        pickle.dump(g, f)
