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

import json

import click
import pandas as pd
from loda import Loda
from util import NetFlowFeatureProcessing


@click.command()
@click.option('--model-name', help="Path to trained LODA IDS model", default="../model/loda_ids.npz")
@click.option('--input-name', help="Input file name path", default="dataset/Friday-WorkingHours-Morning.pcap_ISCX.csv")
@click.option('--config-path', help="Path to JSON training configuration file", default="../model/config.json")
@click.option('--output-name', help="path to result output", default="out.csv")
def inference(input_name, model_name, config_path, output_name):
    # Load config file and apply feature processing
    config = json.load(open(config_path, "r"))
    feature_processor = NetFlowFeatureProcessing(input_name, config=config)
    X_train, _ = feature_processor.process(apply_pca=config['apply_pca'])

    # Load trained Loda model
    model = Loda.load_model(model_name)

    # score model
    scores = model.score(X_train)

    # Result
    df = pd.DataFrame()
    df['scores'] = scores.get()
    df.to_csv(output_name, index=False)


if __name__ == "__main__":

    inference()
