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

# Command to run inference from trained model and saved graph structure.
# python inference.py --input_data dataset/azure_synthetic/azure_ad_logs_sample_with_anomaly_all.json
# --model-dir modeldir/
# --target-node authentication
# --output result.csv

import argparse

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
        model (HeteroRGCN) : trained HeteroRGCN model.
        g (DGLHeterograph) : test graph
        feature_tensors (torch.Tensor) : node features
        test_idx (list): test index
        target_node (list): target node

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


def main():

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

    _, test_data, _, test_idx, labels, df = synthetic_azure(args.input_data)

    g_test, feature_tensors = build_azure_graph(df, meta_cols)

    # Load graph model.
    model, g_training = load_model(args.model_dir)
    model = model.to(device)
    g_test = g_test.to(device)
    test_logits, test_seeds, test_embedding = inference(model, g_test, feature_tensors, test_idx, target_node)

    # collect result
    authentication_score = test_logits[:, 1].numpy()
    df_result = pd.DataFrame(test_embedding.numpy())
    df_result['score'] = authentication_score
    df_result['test_index'] = test_seeds

    # output to csv file with embedding & last two column score, test index
    df_result.to_csv(args.output, index=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-data",
                        help="JSON azure file input",
                        default="dataset/azure_synthetic/azure_ad_logs_sample_with_anomaly_all.json")
    parser.add_argument("--model-dir", help="directory for model files", default="modeldir/")
    parser.add_argument("--target-node", help="Target node", default="authentication")
    parser.add_argument("--output", required=False, help="output filename", default="result.csv")

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    target_node = args.target_node

    main()
