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


import cudf
import cupy as cp
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import os
import glob
from cudf.core.window.rolling import Rolling
from cuml.preprocessing import StandardScaler
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import pickle

import cupy as cp
from cupyx.scipy import sparse


def compute_node_embeddings(edge_index, num_nodes, embedding_dim, node_names, projection_matrix, num_iterations=3,
                            distance_weights=None):
    """
    Compute node embeddings using IPv4 addresses as initial embeddings,
    with distance-based weighting over multiple iterations.

    Args:
        edge_index (cp.ndarray): Edge indices of shape (2, num_edges).
        num_nodes (int): Number of nodes in the graph.
        embedding_dim (int): Desired dimension of the embeddings.
        node_names (dict): Dictionary mapping node indices to IPv4 address strings.
        num_iterations (int): Number of iterations (k).
        distance_weights (list or None): Weights for each distance from 0 to k.
                                         If None, default weights are used.

    Returns:
        cp.ndarray: Node embeddings of shape (num_nodes, embedding_dim).
    """
    # Ensure edge_index is a CuPy array
    if not isinstance(edge_index, cp.ndarray):
        edge_index = cp.asarray(edge_index)

    # Extract rows and columns
    rows = edge_index[0].flatten()
    cols = edge_index[1].flatten()

    # Create data array
    data = cp.ones(len(rows), dtype=cp.float32)

    # Create adjacency matrix A
    A = sparse.coo_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes)).tocsr()

    # Compute inverse degree matrix D_inv
    degrees = cp.array(A.sum(axis=1)).flatten()
    D_inv = cp.where(degrees != 0, 1.0 / degrees, 0.0)
    D_inv_mat = sparse.diags(D_inv)

    # Compute normalized adjacency matrix: D_inv_A = D_inv * A
    D_inv_A = D_inv_mat.dot(A)

    # Step 1: Vectorized processing of IP addresses

    # Create a list of IP addresses aligned with node indices
    ip_addresses = [''] * num_nodes
    for node_idx, ip_address in node_names.items():
        ip_addresses[node_idx] = ip_address

    # Replace empty strings with '0.0.0.0' or another default IP
    ip_addresses_list = ['0.0.0.0' if ip == '' else ip for ip in ip_addresses]

    # Convert to NumPy array
    ip_addresses_np = np.array(ip_addresses_list)

    # Use np.char methods to split IP addresses into octets
    octets = np.char.split(ip_addresses_np, sep='.')

    # Function to pad or truncate octets to length 4
    def pad_or_truncate(lst):
        if len(lst) == 4:
            return lst
        elif len(lst) < 4:
            return lst + ['0'] * (4 - len(lst))
        else:
            return lst[:4]

    # Apply the function vectorized over the array
    octets_padded = np.array([pad_or_truncate(o) for o in octets])

    # Flatten and check if entries are digits
    octets_flat = octets_padded.flatten()
    is_digit = np.char.isdigit(octets_flat)

    # Replace non-digit entries with '0'
    octets_flat[~is_digit] = '0'

    # Convert to integers
    octets_int_flat = octets_flat.astype(np.int32)

    # Reshape back to (num_nodes, 4)
    octets_int = octets_int_flat.reshape((num_nodes, 4))

    # Normalize by dividing by 255.0
    ip_embeddings_np = octets_int.astype(np.float32) / 255.0

    # Convert to CuPy array
    ip_embeddings = cp.asarray(ip_embeddings_np)

    # Optional: Project IP embeddings to the desired embedding dimension
    if embedding_dim != 4:
        # Initialize a projection matrix
        # cp.random.seed(42)
        # projection_matrix = cp.random.uniform(-1, 1, size=(4, embedding_dim)).astype(cp.float32)

        # Project the embeddings
        e0 = ip_embeddings.dot(projection_matrix)
    else:
        e0 = ip_embeddings

    # Normalize initial embeddings
    norms = cp.linalg.norm(e0, axis=1, keepdims=True)
    norms = cp.where(norms == 0, 1.0, norms)  # Prevent division by zero
    e0_normalized = e0 / norms

    # If distance_weights is None, define default weights inversely proportional to distance
    if distance_weights is None:
        # Distance 0 to num_iterations
        distance_weights = [1.0 / (d + 1) for d in range(num_iterations + 1)]
    else:
        assert len(distance_weights) == num_iterations + 1, "Length of distance_weights must be num_iterations + 1"

    # Precompute powers of D_inv_A
    D_inv_A_powers = [sparse.identity(num_nodes, format='csr'), D_inv_A]
    for _ in range(2, num_iterations + 1):
        D_inv_A_powers.append(D_inv_A_powers[-1].dot(D_inv_A))

    # Initialize tensor to hold embeddings at each distance
    e_all = cp.zeros((num_iterations + 1, num_nodes, embedding_dim), dtype=cp.float32)

    # Compute embeddings for distance 0 (initial embeddings)
    e_all[0] = e0_normalized * distance_weights[0]

    # Compute embeddings for distances 1 to num_iterations
    for d in range(1, num_iterations + 1):
        # Compute influence from nodes at distance d
        e_d = D_inv_A_powers[d].dot(e0_normalized)

        # Subtract influence from closer distances to isolate distance d
        for i in range(1, d):
            e_d -= D_inv_A_powers[d - i].dot(e_all[i])

        # Normalize embeddings
        norms = cp.linalg.norm(e_d, axis=1, keepdims=True)
        norms = cp.where(norms == 0, 1.0, norms)
        e_d_normalized = e_d / norms

        # Multiply by weight and store
        e_all[d] = e_d_normalized * distance_weights[d]

    # Sum over distances to get final embeddings
    e_final = cp.sum(e_all, axis=0)

    return e_final


def create_random_tensor(n, m, min_value, max_value):
    # Generate a tensor of shape (n, m) with values between 0 and 1
    random_tensor = torch.rand(n, m, dtype=torch.float32)

    # Scale and shift the values to be within the range [min_value, max_value]
    scaled_tensor = random_tensor * (max_value - min_value) + min_value

    return scaled_tensor


class NetflowPreprocessor:

    def __init__(self, df: cudf.DataFrame, edge_columns: [str], node_dim: int, normalize=True, src_ip='IPV4_SRC_ADDR',
                 dst_ip='IPV4_DST_ADDR', label='Label'):
        self.edge_columns = edge_columns
        self.node_dim = node_dim
        self.edge_scaler = None
        self.scale = normalize
        self.src_ip = src_ip
        self.dst_ip = dst_ip
        self.label = label
        cp.random.seed(42)
        self.projection_matrix = cp.random.uniform(-1, 1, size=(4, self.node_dim)).astype(cp.float32)
        # Assign the dataframe directly
        self.df = df.dropna()

        if self.scale:
            # initialize edge scaler
            self.edge_scaler = StandardScaler().fit(self.df[self.edge_columns].values.get())

    def _generate_windows(self, df, window_size, step_size):
        """ Generate row-based windows instead of time-based ones. """
        windows = []
        total_rows = len(df)
        for start in range(0, total_rows - window_size + 1, step_size):
            windows.append((start, start + window_size))
        return windows

    def save_scaler(self, path='edge_scaler.pkl'):

        assert self.edge_scaler is not None, "Scaler must be initialized to save"

        with open(path, 'wb') as f:
            pickle.dump(self.edge_scaler, f)

    def load_scaler(self, path='edge_scaler.pkl'):

        with open(path, 'rb') as f:
            self.edge_scaler = pickle.load(f)

        self.scale = True

    def process_single(self, df):
        """ Converts a given CuDF dataframe into a single graph. """

        ## Construct node indices
        unique_ips = cudf.concat([df[self.src_ip], df[self.dst_ip]]).unique()
        ip_to_idx = {ip: idx for idx, ip in enumerate(unique_ips.to_pandas())}
        idx_to_ip = {value: key for key, value in ip_to_idx.items()}
        src_idx = df[self.src_ip].map(ip_to_idx).astype(cp.int64)
        dst_idx = df[self.dst_ip].map(ip_to_idx).astype(cp.int64)
        edge_index = cp.vstack([src_idx, dst_idx])

        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(f"edge_index has incorrect shape: {edge_index.shape}")

        # Get and scale edge features
        edge_features = df[self.edge_columns].to_cupy()
        edge_labels = cp.zeros(len(df))  # Dummy feature array

        if self.scale:
            edge_features = cp.asarray(self.edge_scaler.transform(edge_features))

        # Create edge attributes tensor
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_features = torch.tensor(edge_features, dtype=torch.float32)
        edge_labels = torch.tensor(edge_labels, dtype=torch.float32).unsqueeze(1)
        edge_attr = torch.cat([edge_features, edge_labels], dim=1)

        num_nodes = len(unique_ips)
        # Compute node embeddings using CuPy
        embeddings = compute_node_embeddings(
            edge_index=edge_index,
            num_nodes=num_nodes,
            embedding_dim=self.node_dim,
            node_names=idx_to_ip,
            projection_matrix=self.projection_matrix,
            num_iterations=3,  # You can adjust the number of iterations
        )

        x = torch.tensor(embeddings.get(), dtype=torch.float32)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        return data, ip_to_idx

    def construct_graph_list(self, df=None, window_size=1000, step_size=500):
        """ Constructs a list of graphs based on window size and step size. """
        if df is None:
            df = self.df

        # Generate row-based windows
        windows = self._generate_windows(df, window_size, step_size)

        # Create a list to store PyTorch Geometric Data objects
        data_list = []
        ip_map = []
        data_windows = []

        # Iterate through each window and generate PyTorch Geometric graph
        for start, end in tqdm(windows, desc='Graph Windows'):
            window_df = df.iloc[start:end]
            data_windows.append(window_df.copy())

            if not window_df.empty:
                # Create a mapping from IPs to node indices
                unique_ips = cudf.concat([window_df[self.src_ip], window_df[self.dst_ip]]).unique()
                ip_to_idx = {ip: idx for idx, ip in enumerate(unique_ips.to_pandas())}
                idx_to_ip = {value: key for key, value in ip_to_idx.items()}
                ip_map.append(ip_to_idx)

                # Create edges and edge labels directly on the GPU using cuDF and CuPy
                src_idx = window_df[self.src_ip].map(ip_to_idx).astype(cp.int64)
                dst_idx = window_df[self.dst_ip].map(ip_to_idx).astype(cp.int64)
                edge_index = cp.vstack([src_idx, dst_idx])

                # Ensure edge_index has shape (2, num_edges)
                if edge_index.ndim != 2 or edge_index.shape[0] != 2:
                    raise ValueError(f"edge_index has incorrect shape: {edge_index.shape}")

                edge_features = window_df[self.edge_columns].to_cupy()
                edge_labels = window_df[self.label].astype(cp.float32)

                # Scale edge features if needed
                if self.scale:
                    edge_features = cp.asarray(self.edge_scaler.transform(edge_features))

                # Convert to PyTorch tensors
                edge_index = torch.tensor(edge_index, dtype=torch.long)
                edge_features = torch.tensor(edge_features, dtype=torch.float32)
                edge_labels = torch.tensor(edge_labels.values, dtype=torch.float32).unsqueeze(1)

                # Combine edge features with labels
                edge_attr = torch.cat([edge_features, edge_labels], dim=1)

                # Use a zero-filled node feature tensor of size (number of nodes, node_dim)
                num_nodes = len(unique_ips)
                # Compute node embeddings using CuPy
                embeddings = compute_node_embeddings(
                    edge_index=edge_index,
                    num_nodes=num_nodes,
                    embedding_dim=self.node_dim,
                    node_names=idx_to_ip,
                    projection_matrix=self.projection_matrix,
                    num_iterations=3,  # You can adjust the number of iterations
                )

                # Convert embeddings to PyTorch tensor
                x = torch.tensor(embeddings.get(), dtype=torch.float32)

                # Create PyTorch Geometric Data object
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
                data_list.append(data)

        return data_list, ip_map, data_windows