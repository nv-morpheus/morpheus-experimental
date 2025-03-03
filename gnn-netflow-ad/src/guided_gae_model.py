# SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, TransformerConv, GAE, GraphUNet


class GATEncoderWithEdgeAttr(nn.Module):
    def __init__(self, in_channels, hidden_channels, edge_attr_dim, num_heads=4):
        super(GATEncoderWithEdgeAttr, self).__init__()
        self.num_heads = num_heads
        self.hidden_channels = hidden_channels

        # First GATv2Conv layer
        self.conv1 = GATv2Conv(
            in_channels,
            hidden_channels,
            heads=num_heads,
            edge_dim=edge_attr_dim,
            concat=True  # Output will be (hidden_channels * num_heads)
        )

        # BatchNorm adjusted to match the output features of conv1
        self.batch_norm = nn.BatchNorm1d(hidden_channels * num_heads)

        self.unet = GraphUNet(
            hidden_channels * num_heads,
            hidden_channels,
            hidden_channels,
            3,
        )

    def forward(self, x, edge_index, edge_attr):
        # First GATv2Conv layer]
        # print(f"Peak memory usage before conv1: {torch.cuda.max_memory_allocated(torch.device('cuda' if torch.cuda.is_available() else 'cpu')) / 1e9} GB")
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        # print(f"Peak memory usage before batch_norm: {torch.cuda.max_memory_allocated(torch.device('cuda' if torch.cuda.is_available() else 'cpu')) / 1e9} GB")
        x = self.batch_norm(x)
        # Second GATv2Conv layer
        # x = self.conv2(x, edge_index, edge_attr)
        # print(f"Peak memory usage before unet: {torch.cuda.max_memory_allocated(torch.device('cuda' if torch.cuda.is_available() else 'cpu')) / 1e9} GB")
        x = self.unet(x, edge_index)
        # print(f"Peak memory usage after unet: {torch.cuda.max_memory_allocated(torch.device('cuda' if torch.cuda.is_available() else 'cpu')) / 1e9} GB")
        return x  # Node embeddings (num_nodes, hidden_channels)


class GlobalEdgeEmbedding(nn.Module):
    def __init__(self, edge_attr_dim, global_emb_dim):
        super(GlobalEdgeEmbedding, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(edge_attr_dim, global_emb_dim),
            nn.ReLU(),
            nn.Linear(global_emb_dim, global_emb_dim)
        )

    def forward(self, edge_attr, edge_batch):
        # Aggregate edge attributes per graph using mean pooling
        # edge_attr: [num_edges, edge_attr_dim]
        # edge_batch: [num_edges], indicating which graph each edge belongs to

        global_edge_attr = global_mean_pool(edge_attr, edge_batch)  # Shape: [batch_size, edge_attr_dim]
        # Compute global embedding for each graph
        global_emb = self.mlp(global_edge_attr)  # Shape: [batch_size, global_emb_dim]
        return global_emb  # Global edge embedding: [batch_size, global_emb_dim]


class DecoderWithGlobalEdge(nn.Module):
    def __init__(self, hidden_channels, edge_attr_dim, global_emb_dim):
        super(DecoderWithGlobalEdge, self).__init__()
        self.hidden_channels = hidden_channels

        self.fc_edge = nn.Linear(edge_attr_dim, hidden_channels)
        self.fc_global = nn.Linear(global_emb_dim, hidden_channels)
        # Adjusted the input dimension based on concatenated features
        self.fc = nn.Linear(3 * hidden_channels, 1)
        self.fc1 = nn.Linear(3 * hidden_channels, 3 * hidden_channels)

    def forward(self, z, edge_index, edge_attr, global_edge_emb, edge_batch):
        row, col = edge_index
        z_row = z[row]  # Shape: [num_edges, hidden_channels]
        z_col = z[col]  # Shape: [num_edges, hidden_channels]

        # Element-wise multiplication for node interactions
        node_interaction = z_row * z_col  # Shape: [num_edges, hidden_channels]

        # Transform local edge attributes
        edge_attr_transformed = F.relu(self.fc_edge(edge_attr))  # Shape: [num_edges, hidden_channels]

        # Get the corresponding global edge embedding for each edge
        global_edge_emb_expanded = global_edge_emb[edge_batch]  # Shape: [num_edges, global_emb_dim]
        global_edge_emb_transformed = F.relu(
            self.fc_global(global_edge_emb_expanded))  # Shape: [num_edges, hidden_channels]

        # Concatenate features including node interaction terms
        concat = torch.cat([node_interaction, edge_attr_transformed, global_edge_emb_transformed],
                           dim=1)  # Shape: [num_edges, 3 * hidden_channels]

        # Predict edge existence with sigmoid activation
        out = torch.sigmoid(self.fc(
            F.elu(
                self.fc1(concat)
            )

        )).squeeze()  # Shape: [num_edges]
        return out  # Edge probabilities (num_edges,)


class GAEWithGlobalEdge(GAE):
    def __init__(self, encoder, decoder, global_edge_embedding):
        super(GAEWithGlobalEdge, self).__init__(encoder=encoder)
        self.decoder = decoder
        self.global_edge_embedding = global_edge_embedding

    def encode(self, x, edge_index, edge_attr):
        return self.encoder(x, edge_index, edge_attr)

    def decode(self, z, edge_index, edge_attr, batch):
        # Create edge_batch from node batch
        edge_batch = batch[edge_index[0]]  # Assuming edge_index[0] corresponds to source nodes

        # Compute global edge embeddings
        global_edge_emb = self.global_edge_embedding(edge_attr, edge_batch)  # Shape: [batch_size, global_emb_dim]

        return self.decoder(z, edge_index, edge_attr, global_edge_emb, edge_batch)




