import os
import datetime
import logging
import numpy as np
import cudf
import cuml
import cuml.preprocessing as cupreproc
import pandas as pd
import pickle
import click
import pdb
from utils import compute_chars

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def normalize_host_data(data_fname_, norm_method='l2', preproc='minmax'):
    df = cudf.read_csv(data_fname_)
    print("Num. of columns:{}".format(len(df.columns)))

    rm_assets = ['ActiveDirectory', 'EnterpriseAppServer']
    print("\nREMOVED {} Assets from data".format(rm_assets))
    df = df.loc[~df['LogHost'].isin(rm_assets)]

    rm_cols = ['LogHost', 'uname_other_compacnt_login_frac', 'uname_that_compacnt_login_frac']
    norm_cols_ = [x for x in df.columns if x not in rm_cols]

    if preproc == 'unit_norm':
        scaler = cupreproc.normalize(norm=norm_method)
    elif preproc == 'minmax':
        scaler = cupreproc.MinMaxScaler(feature_range=(0, 1))

    df_norm = scaler.fit(df[norm_cols_]).transform(df[norm_cols_])
    df_norm.columns = norm_cols_

    return df, df_norm


@click.command()
@click.option('--model', default='dbscan', help='Clustering method to use.'\
     ' Valid choices are \'kmeans\' or \'dbscan\'. Default is \'dbscan\'')
def run(**kwargs):
    model = kwargs['model']
    compute_cluster_chars = True
    assert model in ['kmeans', 'dbscan']

    global NUM_DAYS, norm_cols
    data_fname = '../datasets/host_agg_data_day-11_day-15.csv'
    NUM_DAYS = 5.0
    df, df_norm = normalize_host_data(data_fname)

    if model=='dbscan':
        fname = '../models/dbscan_eps0.0005.pkl'
        clust_ = "cluster_dbscan_eps0.0005_minkp1"

        dbsc_model, pca, pca_dims = pickle.load(open(fname, "rb"))
        df_pca = pca.transform(df_norm).iloc[:,:pca_dims]
        df[clust_] = dbsc_model.fit_predict(df_pca)

    elif model=='kmeans':
        fname = "../models/kmeans_16clusts.pkl"
        clust_ = "cluster_KM_16"

        kmeans_model, pca, pca_dims = pickle.load(open(fname, "rb"))
        df_pca = pca.transform(df_norm).iloc[:,:pca_dims]
        df[clust_] = kmeans_model.predict(df_pca)

    print("Cluster Size:\n{}".format(df[clust_].value_counts()))

    if compute_cluster_chars:
        cluster_chars = compute_chars(df, clust_, cluster_id=0, NUM_DAYS=NUM_DAYS)

    return


if __name__ == '__main__':
    dt = datetime.date.today()
    logger_fname = 'logs/inference.log'.format(dt.strftime('%d%m%y'))
    print("Logging in {}".format(logger_fname))
    logging.basicConfig(level=logging.DEBUG, filename=logger_fname,
                        filemode='a', format='%(asctime)s: %(message)s',
                        datefmt='%m%d-%H%M')
    run()