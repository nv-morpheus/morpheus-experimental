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

import json

import dgl
import pandas as pd
import torch


def map_node_id(df, col_name):
    """ Convert column node list to integer index for dgl graph.

    Args:
        df (DataFrame): dataframe
        col_name (list: column list
    """
    node_index = {j: i for i, j in enumerate(df[col_name].unique())}
    df[col_name + "_id"] = df[col_name].map(node_index)


def build_azure_graph(train_data, col_drop):
    """Build heterograph from edglist and node index.

    Args:
        train_data (_type_): training data for node features.
        col_drop (_type_): features to drop from node features.

    Returns:
        _type_: dlg graph, normalized feature tensor
    """

    edge_list = {
        ('user', 'requested', 'authentication'): (train_data['userId_id'].values, train_data['auth_id'].values),
        ('authentication', 'recieve', 'user'): (train_data['auth_id'].values, train_data['userId_id'].values),
        ('authentication', 'access', 'app'): (train_data['auth_id'].values, train_data['appId_id'].values),
        ('app', 'accessed-by', 'authentication'): (train_data['appId_id'].values, train_data['auth_id'].values),
        ('device', 'request', 'authentication'): (train_data['ipAddress_id'].values, train_data['auth_id'].values),
        ('authentication', 'request-by', 'device'): (train_data['auth_id'].values, train_data['ipAddress_id'].values)
    }
    G = dgl.heterograph(edge_list)
    feature_tensors = torch.tensor(train_data.drop(col_drop, axis=1).values).float()
    feature_tensors = (feature_tensors - feature_tensors.mean(0)) / (0.0001 + feature_tensors.std(0))

    return G, feature_tensors


def prepare_data(df_cleaned):
    """ Prepare aggregated features from raw azure dataframe.

    Args:
        df_cleaned (DataFrame):raw azure dataset converted from json.

    Returns:
        _type_: feature processed dataframe
    """

    # convert bool features to int and set status_flag label based on succcess error code.
    df_cleaned['riskDetail'] = (df_cleaned['riskDetail'] == 'none').astype(int)
    df_cleaned['deviceDetail.isCompliant'] = (~df_cleaned['deviceDetail.isCompliant'].isna()).astype(int)
    df_cleaned['deviceDetail.isManaged'] = (~df_cleaned['deviceDetail.isManaged'].isna()).astype(int)
    # df_cleaned['status_flag'] = (df_cleaned['status.failureReason'] != 'Other.').astype(int)
    df_cleaned['status_flag'] = (df_cleaned['status.errorCode'] != 0).astype(int)

    # Create OHE set features & their aggregation function.
    ohe_cols = [
        'deviceDetail.trustType',
        'riskState',
        'riskLevelAggregated',
        'riskLevelDuringSignIn',
        'clientAppUsed',
        'deviceDetail.operatingSystem'
    ]

    df_ohe = pd.get_dummies(df_cleaned, columns=ohe_cols, prefix=ohe_cols, prefix_sep='_')
    ohe_col_agg = {c: 'sum' for col in ohe_cols for c in df_ohe.columns if c.startswith(col)}

    agg_func = {
        'location.city': 'nunique',
        'location.countryOrRegion': 'nunique',
        'resourceDisplayName': 'nunique',
        'fraud_label': 'max',
        'deviceDetail.isCompliant': 'sum',
        'deviceDetail.isManaged': 'sum',
        'riskDetail': 'sum',
        'status_flag': 'max'
    }

    agg_func = {**agg_func, **ohe_col_agg}
    group_by = ['appId', 'userId', 'ipAddress', 'day']
    grouped_df = df_ohe.groupby(group_by).agg(agg_func).reset_index()

    return grouped_df


def convert_json_csv_schema(json_df):
    """Convert raw json azure to model input dataframe.

    Args:
        json_df (_type_): json input dataset

    Returns:
        _type_: dataframe
    """

    feature_list = [
        '_time',
        'appId',
        'clientAppUsed',
        'createdDateTime',
        'date_hour',
        'deviceDetail.browser',
        'deviceDetail.deviceId',
        'deviceDetail.displayName',
        'deviceDetail.isCompliant',
        'deviceDetail.isManaged',
        'deviceDetail.operatingSystem',
        'deviceDetail.trustType',
        'id',
        'ipAddress',
        'isInteractive',
        'location.city',
        'location.countryOrRegion',
        'location.state',
        'resourceDisplayName',
        'riskDetail',
        'riskLevelAggregated',
        'riskLevelDuringSignIn',
        'riskState',
        'status.failureReason',
        'status.errorCode',
        'userId',
        'userPrincipalName',
        'day'
    ]

    new_schema_features = sorted(json_df.columns.tolist())
    col_csv_json = {col: None for col in feature_list}
    for csv_col in sorted(feature_list):
        for jsc in new_schema_features:
            if jsc.startswith('prop') and jsc[11:] == csv_col:
                col_csv_json[csv_col] = jsc
            elif jsc == csv_col:
                col_csv_json[csv_col] = jsc
    col_csv_json['_time'] = 'time'
    col_json_csv = {json: csv for csv, json in col_csv_json.items() if json}

    csv_df = json_df[col_json_csv.keys()]
    csv_df = csv_df.rename(columns=col_json_csv)
    csv_df['_time'] = pd.to_datetime(csv_df['_time'])
    csv_df['day'] = csv_df._time.dt.dayofyear

    return csv_df


def get_fraud_label_index(df):
    fraud_index = (df['_time'].dt.day >= 30) & (df['userPrincipalName'] == 'attacktarget@domain.com')
    return fraud_index


def synthetic_azure(file_name, split_day=241):
    """Process input json azure file and produce processed training and test data

    Args:
        file_name (str): json file input name.
        split_day (int): day split for training & test dataset. Default 241

    Returns:
        _type_: training_data, train_index, test_index, test_data, training_label, original data
    """
    # Load Json, convert to dataframe, extract features.
    df = pd.json_normalize(json.load(open(file_name, 'r')))
    df = convert_json_csv_schema(df)

    # set fraud index
    fraud_index = get_fraud_label_index(df)
    df['fraud_label'] = 0.0
    df.loc[fraud_index, 'fraud_label'] = 1.0

    df = prepare_data(df)

    # map node to index
    df['auth_id'] = df.index
    for col in ['appId', 'userId', 'ipAddress']:
        map_node_id(df, col)

    # split data into training and test based on days.
    test_mask = (df.day > split_day)
    train_data = df[~test_mask]
    test_data = df[test_mask]
    return train_data, test_data, train_data.index, test_data.index, df['status_flag'].values, df
