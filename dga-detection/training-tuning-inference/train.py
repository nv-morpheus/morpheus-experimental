# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""
Example Usage:
python train.py \
       --training-data ../datasets/benign_and_dga_domains.csv
"""

import argparse
import os
from datetime import datetime

from dga_detector import DGADetector

import cudf

N_LAYERS = 3
CHAR_VOCAB = 128
HIDDEN_SIZE = 100
N_DOMAIN_TYPE = 2
LR = 0.001
EPOCHS = 25
TRAIN_SIZE = 0.7
BATCH_SIZE = 10000
MODELS_DIR = 'models'


def main():
    print("Load Input Dataset to GPU Dataframe...")
    gdf = cudf.read_csv(args.training_data)
    train_data = gdf['domain']
    labels = gdf['type']

    print("Instantiate DGA detector...")
    dd = DGADetector(lr=LR)
    dd.init_model(n_layers=N_LAYERS, char_vocab=CHAR_VOCAB, hidden_size=HIDDEN_SIZE, n_domain_type=N_DOMAIN_TYPE)

    print("Model Training and Evaluation...")
    dd.train_model(train_data, labels, batch_size=BATCH_SIZE, epochs=EPOCHS, train_size=0.7)

    print("Saving Model...")
    if not os.path.exists(MODELS_DIR):
        print("Creating directory '{}'".format(MODELS_DIR))
        os.makedirs(MODELS_DIR)

    now = datetime.now()
    model_filename = "rnn_classifier_{}.bin".format(now.strftime("%Y-%m-%d_%H_%M_%S"))
    model_filepath = os.path.join(MODELS_DIR, model_filename)
    dd.save_checkpoint(model_filepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-data", required=True, help="CSV with `domain` and `type` with 0/1 label")
    args = parser.parse_args()

main()
