import dgl
import pandas as pd
import numpy as np
import torch
import json


def map_node_id(df, col_name):
    """ map node to id.

    Args:
        df (_type_): dataframe
        col_name (_type_): column list
    """
    node_index = {j: i for i, j in enumerate(df[col_name].unique())}
    df[col_name+"_id"] = df[col_name].map(node_index)


def build_azure_graph(train_data, col_drop):
    """Build graph 

    Args:
        train_data (_type_): _description_
        col_drop (_type_): _description_

    Returns:
        _type_: dlg graph, feature tensor
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
    feature_tensors = torch.tensor(
        train_data.drop(
            col_drop,
            axis=1).values).float()
    feature_tensors = (feature_tensors - feature_tensors.mean(0)
                       ) / (0.0001 + feature_tensors.std(0))

    return G, feature_tensors


def get_anonomized_dataset():
    """Return anonomized dataset. 

    Returns:
        _type_: _description_
    """

    df = pd.read_parquet('data/anomized_azure.pq')
    test_mask = (df.day > 43) & (df.day < 60)
    train_data = df[~test_mask]
    test_data = df[test_mask]

    return train_data, test_data, train_data.index, test_data.index, df[
        'status_flag'].values, df


def prepare_data(df_cleaned):
    
    df_cleaned['riskDetail'] = (df_cleaned['riskDetail'] == 'none').astype(int)
    df_cleaned['deviceDetail.isCompliant'] = (df_cleaned['deviceDetail.isCompliant'] == True).astype(int)
    df_cleaned['deviceDetail.isManaged'] = (df_cleaned['deviceDetail.isManaged'] == True).astype(int)
    df_cleaned['status_flag'] = (df_cleaned['status.failureReason'] != 'Other.').astype(int)

    # Grouping based on selected features. 
    count_cats = pd.Series({cc: df_cleaned[cc].nunique() for cc in df_cleaned.columns}).sort_values()
    # Filter specific columns to apply OHE:
    #ohe_cols = list(count_cats[count_cats.between(3,60)].index)
    #ohe_cols.remove('status.failureReason')
    ohe_cols = ['deviceDetail.trustType', 'riskState', 'riskLevelAggregated', 'riskLevelDuringSignIn', 'clientAppUsed',                                               'deviceDetail.operatingSystem']

    df_ohe = pd.get_dummies(df_cleaned, columns=ohe_cols, prefix=ohe_cols, prefix_sep='_')
    ohe_col_agg = { c:'sum' for col in ohe_cols for c in df_ohe.columns if c.startswith(col) }
    
    print(ohe_cols)
    agg_func = {
    'location.city':'nunique',
    'location.countryOrRegion':'nunique',
    'resourceDisplayName':'nunique',
    'fraud_label': 'max',
    'deviceDetail.isCompliant': 'sum',
    'deviceDetail.isManaged': 'sum',
    'riskDetail': 'sum',
    'status_flag': 'max'
   
    }
    df_ohe['fraud_label'] = 0.0
    #df_ohe['fraud_label'][get_fraud_index(df)] = 1.0
    agg_func = {**agg_func, **ohe_col_agg}
    group_by = ['appId','userId','ipAddress','day']
    print(df_ohe.columns)
    gg = df_ohe.groupby(group_by).agg(agg_func).reset_index()

    return gg

  



def convert_json_csv_schema(json_df):

    column_list_sel = [
        '_time', 'appId', 'clientAppUsed',
        'createdDateTime', 'date_hour', 'deviceDetail.browser',
        'deviceDetail.deviceId', 'deviceDetail.displayName',
        'deviceDetail.isCompliant', 'deviceDetail.isManaged',
        'deviceDetail.operatingSystem', 'deviceDetail.trustType', 'id',
        'ipAddress', 'isInteractive',  'location.city',
        'location.countryOrRegion', 'location.state', 'resourceDisplayName',
        'riskDetail', 'riskLevelAggregated', 'riskLevelDuringSignIn',
        'riskState',  'status.failureReason',
        'userId', 'userPrincipalName', 'day']

    new_schema_column = sorted(json_df.columns.tolist())
    col_csv_json = {col: None for col in column_list_sel}
    for csv_col in sorted(column_list_sel):
        for jsc in new_schema_column:
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


def synthetic_azure(file_name):
    # Load raw json dataset
    df = pd.json_normalize(json.load(open(file_name, 'r')))
    df = convert_json_csv_schema(df)
    df = prepare_data(df)
    
    test_mask = (df.day>330) #(grp.day > 43) & (grp.day < 60)
    train_data = df[~test_mask]
    test_data = df[test_mask]
    return train_data, test_data, train_data.index, test_data.index, df[
        'status_flag'].values, df


def azure_data():
    # preprocessed data
    # Load raw dataset
    # feature engineering & aggregation
    return get_anonomized_dataset()
