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

import cupy
import cudf


def combine_preds(edge_index, ip_to_idx, preds, df):
    # Step 1: Map 'src_ip' and 'dst_ip' to their corresponding node indices
    ip_series = cudf.Series(ip_to_idx)
    df['src_node_idx'] = df['IPV4_SRC_ADDR'].map(ip_series)
    df['dst_node_idx'] = df['IPV4_DST_ADDR'].map(ip_series)

    # Ensure node indices are CuPy arrays of int64 to prevent overflow
    df_src_node_idx = df['src_node_idx'].values.astype(cupy.int64)
    df_dst_node_idx = df['dst_node_idx'].values.astype(cupy.int64)

    # Handle missing node indices if necessary
    # For this example, we'll assume all IPs are in ip_to_idx

    # Step 2: Compute a composite key for edges in edge_index
    # Convert node indices to int64
    edge_src_node_idx = edge_index[0].astype(cupy.int64)
    edge_dst_node_idx = edge_index[1].astype(cupy.int64)

    # Calculate the maximum node index
    N = cupy.int64(max(
        edge_src_node_idx.max().get(),
        edge_dst_node_idx.max().get(),
        df_src_node_idx.max().get(),
        df_dst_node_idx.max().get()
    )) + 1

    # Compute composite keys for edges in edge_index
    edge_keys = edge_src_node_idx * N + edge_dst_node_idx

    # Sort edge_keys and predictions accordingly
    sorted_indices = cupy.argsort(edge_keys)
    edge_keys_sorted = edge_keys[sorted_indices]
    preds_sorted = preds[sorted_indices]

    # Step 3: Compute composite keys for edges in df
    df_edge_keys = df_src_node_idx * N + df_dst_node_idx

    # Use searchsorted to find indices where df_edge_keys would fit in edge_keys_sorted
    indices = cupy.searchsorted(edge_keys_sorted, df_edge_keys, side='left')

    # Adjust indices that are out of bounds
    indices[indices >= len(edge_keys_sorted)] = len(edge_keys_sorted) - 1

    # Step 4: Verify if the keys at found indices match df_edge_keys
    matched = edge_keys_sorted[indices] == df_edge_keys

    # Initialize predictions with NaN (or any default value you prefer)
    df_preds = cupy.full(len(df), cupy.nan, dtype=preds.dtype)

    # Assign predictions where matches are found
    df_preds[matched] = preds_sorted[indices[matched]]

    df['probability_anomalous'] = df_preds

    return df

