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

import argparse

import datatools
import model
import pandas as pd
import torch


def main(file_name, model_name):

    # Read input data
    df = pd.read_csv(file_name)

    # Load trained model and parameters
    check_point = torch.load(model_name)

    window_df = datatools.preprocess(df, check_point['W2V_conf']['WINDOW_SIZE'], check_point['W2V_conf']['STEP_SIZE'])

    # convert to input vector
    test_vector = datatools.test_vector(window_df,
                                        check_point['W2V_conf']['train_dict'],
                                        check_point['W2V_conf']['w2v_dict'])

    # load LogLSTM model
    trained_model_ = model.LogLSTM(**check_point['model_hyperparam']).to(device)
    trained_model_.load_state_dict(check_point['model_state_dict'])

    # predict label
    _, y_pred = model.model_inference(trained_model_, device, test_vector['W2V_EventId'].values.tolist())

    # collect result
    df_result = test_vector
    df_result['prediction'] = y_pred

    # output to csv file with embedding & last two column score, test index
    df_result.to_csv(args.output, index=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-data", help="CSV log file", default="dataset/BGL_2k.log_structured.csv")
    parser.add_argument("--model-name", help="directory for model files", default="model/model_BGL.pt")
    parser.add_argument("--output", required=False, help="output filename", default="result.csv")

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    main(args.input_data, args.model_name)
