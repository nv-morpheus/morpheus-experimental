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

"""
Example Usage:
python ot-xgboost-inference.py \
    --model ../models/ot-xgboost-20230207.pkl \
    --output ot-validation-output.jsonlines
"""

import glob
import requests
import pandas as pd
import os.path
import numpy as np
import subprocess
import pickle
import argparse
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


def inference(model, output):

    # Download the dataset

    if not os.path.isfile("triple.7z"):

        URL = "http://www.ece.uah.edu/~thm0009/icsdatasets/triple.7z"
        response = requests.get(URL)
        open("triple.7z", "wb").write(response.content)

    # Unzip the dataset

    if not os.path.isfile("data1.csv"):

        subprocess.run(['p7zip', '-k', '-d', 'triple.7z'],
                       stdout=subprocess.PIPE)

    # Read the data into a dataset and save a copy of the merged dataframe

    if not os.path.isfile("3class.csv"):
        all_files = glob.glob(os.path.join("*.csv"))

        dflist = []
        for i in all_files:
            dflist.append(pd.read_csv(i))
        df = pd.concat(dflist)
        df.reset_index(drop=True, inplace=True)

    else:
        df = pd.read_csv("3class.csv")

    # Replace infinite values with nan

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Replace labels with numbers
    df["marker"] = df["marker"].replace("NoEvents", 0)
    df["marker"] = df["marker"].replace("Attack", 1)
    df["marker"] = df["marker"].replace("Natural", 2)

    # Replace the nan values with the median of each column.

    df = df.fillna(df.median())

    # Create dataframes for input and labels.

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Create train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # Start an XGBoost classifier

    xgb_clf = XGBClassifier()
    with open(model, "rb") as file:
        xgb_clf = pickle.load(file)

    # Use the loaded model for predictions

    y_pred = xgb_clf.predict(X_test)

    f1 = f1_score(y_test, y_pred, average="weighted")

    print("F1 score is ", f1)
    X_test["predictions"] = y_pred
    X_test.to_json(output, orient='records', lines=True)


def main():

    inference(args.model, args.output)
    print("Inference completed, output saved")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="trained model")
    parser.add_argument("--output", required=True, help="output filename")
    args = parser.parse_args()

main()
