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

import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F


class HeteroRGCNLayer(nn.Module):
    """Relational graph convolutional layer

    Args:
        in_size (int): input feature size.
        out_size (int): output feature size.
        etypes (list): edge relation names.
    """

    def __init__(self, in_size, out_size, etypes):
        super(HeteroRGCNLayer, self).__init__()
        # W_r for each relation
        input_sizes = [in_size] * len(etypes) if type(in_size) == int else in_size
        self.weight = nn.ModuleDict({name: nn.Linear(in_dim, out_size) for name, in_dim in zip(etypes, input_sizes)})

    def forward(self, G, feat_dict):
        """Forward computation

        Args:
            G (DGLHeterograph): Input graph
            feat_dict (dict[str, torch.Tensor]): Node features for each node.

        Returns:
            dict[str, torch.Tensor]: New node features for each node type.
        """
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
    """Relational graph convolutional layer

    Args:
        g (DGLHeterograph): input graph.
        in_size (int): input feature size.
        hidden_size (int): hidden layer size.
        out_size (int): output feature size.
        n_layers (int): number of layers.
        embedding_size (int): embedding size
        device (str, optional): host device. Defaults to 'cpu'.
        target (str, optional): target node. Defaults to 'authentication'.
    """

    def __init__(self,
                 g,
                 in_size,
                 hidden_size,
                 out_size,
                 n_layers,
                 embedding_size,
                 device='cpu',
                 target='authentication'):

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
        self.device = device
        self.g_embed = None

        # create layers
        in_sizes = [in_size if src_type == self.target else embedding_size for src_type, _, _ in g.canonical_etypes]
        layers = [HeteroRGCNLayer(in_sizes, hidden_size, g.etypes)]

        # hidden layers
        for i in range(n_layers - 1):
            layers.append(HeteroRGCNLayer(hidden_size, hidden_size, g.etypes))

        # output layer
        layers.append(nn.Linear(hidden_size, out_size))
        self.layers = nn.Sequential(*layers)

    def embed(self, g, features):
        """Embeddings for all node types.

        Args:
            g (DGLHeterograph): Input graph
            features (torch.Tensor): target node features

        Returns:
            list: target node embedding
        """
        # get embeddings for all node types. Initialize nodes with random weight.
        h_dict = {self.target: features}
        for ntype in self.embed_dict:
            if g[0].number_of_nodes(ntype) > 0:
                h_dict[ntype] = self.embed_dict[ntype][g[0].nodes(ntype).to(self.device)]

        # Forward pass to layers.
        for i, layer in enumerate(self.layers[:-1]):
            if i != 0:
                h_dict = {k: F.leaky_relu(h) for k, h in h_dict.items()}
            h_dict = layer(g[i], h_dict)
        self.g_embed = h_dict
        return h_dict[self.target]

    def forward(self, g, features):
        """Perform forward inference on graph G with feature tensor input

        Args:
            g (DGLHeterograph): DGL test graph
            features (torch.Tensor): input feature
        Returns:
            list: layer embedding
        """
        return self.layers[-1](self.embed(g, features))

    def infer(self, g, features):
        """Perform forward inference on graph G with feature tensor input

        Args:
            g (DGLHeterograph): DGL test graph
            features (torch.Tensor): input feature

        Returns:
            list: logits, embedding vector
        """
        embedding = self.embed(g, features)
        predictions = self.layers[-1](embedding)
        return nn.Sigmoid()(predictions), embedding
