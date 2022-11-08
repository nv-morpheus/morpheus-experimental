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

# Inference from trained model and saved graph structure.

import dgl
import pandas as pd
import torch
from data_processing import build_azure_graph
from data_processing import synthetic_azure
from model_training import evaluate
from model_training import load_model


def inference(model, g, feature_tensors, test_idx, target_node):
    """Minibatch inference on test graph

    Args:
        model : trained HeteroRGCN
        g : test graph
        feature_tensors : node features
        test_idx : test index
        target_node : target node

    Returns:
        list: logits, index, output embedding
    """

    # create sampler and test dataloaders
    full_sampler = dgl.dataloading.MultiLayerNeighborSampler([4, 3])
    test_dataloader = dgl.dataloading.NodeDataLoader(g, {target_node: test_idx},
                                                     full_sampler,
                                                     batch_size=100,
                                                     shuffle=False,
                                                     drop_last=False,
                                                     num_workers=0)
    test_logits, test_seeds, test_embedding = evaluate(model, test_dataloader, feature_tensors, target_node)

    return test_logits, test_seeds, test_embedding


if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    target_node = "authentication"

    # Read input data
    meta_cols = [
        'day',
        'appId',
        'userId',
        'ipAddress',
        'fraud_label',
        'appId_id',
        'userId_id',
        'ipAddress_id',
        'auth_id',
        'status_flag'
    ]

    _, test_data, _, test_idx, labels, df = synthetic_azure(
        'dataset/azure_synthetic/azure_ad_logs_sample_with_anomaly_train.json')

    g_test, feature_tensors = build_azure_graph(df, meta_cols)

    # Load graph model.
    model, g_training = load_model("modeldir/")
    model = model.to(device)
    g_test = g_test.to(device)
    test_logits, test_seeds, test_embedding = inference(model, g_test, feature_tensors, test_idx, target_node)

    # collect result
    authentication_score = test_logits[:, 1].numpy()
    df_result = pd.DataFrame(test_embedding.numpy())
    df_result['score'] = authentication_score
    df_result['test_index'] = test_seeds

    # output to csv file with embedding & last two column score, test index
    df_result.to_csv('result.csv', index=False)
