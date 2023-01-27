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

import datetime
import logging
import pickle

import click
from utils import compute_chars
from utils import normalize_host_data


@click.command()
@click.option('--model', default='dbscan', help='Clustering method to use.'\
    ' Valid choices are \'kmeans\' or \'dbscan\'. Default is \'dbscan\'.'\
    'The corresponding model pickle file will be read from the relative'\
    'path \'../models/ \'.')
@click.option('--data_fname', default='host_agg_data_day-11_day-15.csv',\
    help='Name of the Preprocessed csv dataset to perofrm inference. The given'\
    'file name will be read from the relative path \'../datasets/ \'')
@click.option('--num_days', default=5.0, help='Number of days worth of data used'\
    'in preparing the dataset. Used to normalize the features.')
@click.option('--compute_cluster_chars', is_flag=True, help='Boolean flag. If '\
    'not provided, script just performs inference and output the cluster sizes.'\
    'If provided, additionally analyzes for the top salient features of each cluster'\
    'and prints the analysis to stdout.')
def run(**kwargs):
    dataset_path = '../datasets/'
    model_path = '../models/'

    model = kwargs['model']
    num_days = kwargs['num_days']
    compute_cluster_chars = kwargs['compute_cluster_chars']
    assert model in ['kmeans', 'dbscan'], \
        "Valid choices for model are kmeans or dbscan"

    data_path = dataset_path + kwargs['data_fname']
    df, df_norm = normalize_host_data(data_path)

    if model == 'dbscan':
        fname = model_path + 'dbscan_eps0.0005.pkl'
        clust = "cluster_dbscan_eps0.0005_minkp1"

        dbsc_model, pca, pca_dims = pickle.load(open(fname, "rb"))
        df_pca = pca.transform(df_norm).iloc[:, :pca_dims]
        df[clust] = dbsc_model.fit_predict(df_pca)

    elif model == 'kmeans':
        fname = model_path + 'kmeans_16clusts.pkl'
        clust = "cluster_KM_16"

        kmeans_model, pca, pca_dims = pickle.load(open(fname, "rb"))
        df_pca = pca.transform(df_norm).iloc[:, :pca_dims]
        df[clust] = kmeans_model.predict(df_pca)

    print("Cluster Size:\n{}".format(df[clust].value_counts()))

    if compute_cluster_chars:
        cluster_chars = compute_chars(df, clust, cluster_id=0, num_days=num_days)

    return


if __name__ == '__main__':
    dt = datetime.date.today()
    logger_fname = 'logs/inference.log'.format(dt.strftime('%d%m%y'))
    print("Logging in {}".format(logger_fname))
    logging.basicConfig(level=logging.DEBUG,
                        filename=logger_fname,
                        filemode='a',
                        format='%(asctime)s: %(message)s',
                        datefmt='%m%d-%H%M')
    run()
