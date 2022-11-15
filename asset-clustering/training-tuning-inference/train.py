import os
import datetime
import logging
import numpy as np
import sklearn.cluster as skcluster
import sklearn.manifold as skmani
import matplotlib.pyplot as plt
import cudf
import cuml
import cuml.preprocessing as cupreproc
from cuml.metrics.cluster import silhouette_score
import pandas as pd
import pickle
import click
import pdb
from utils import compute_chars

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def train(df_, model):
    model.fit(df_)
    return model


def get_kmeans(n_clusters_=5):
    kmeans_model = cuml.KMeans(n_clusters=n_clusters_,
                                    init='scalable-k-means++',
                                    n_init=3, max_iter=300,
                                    tol=0.0001, verbose=0)
    return kmeans_model


def iterate_kmeans(df_, verbose=True):
    labels = cudf.DataFrame()
    inertia_dict = {}
    for n_clusters_ in range(2, 30, 2):
        model = get_kmeans(n_clusters_=n_clusters_)
        model = train(df_, model)
        inertia_dict[n_clusters_] = model.inertia_
        labels['KMeans_'+str(n_clusters_)] = model.predict(df_)
        if verbose:
            print("Clusters:{}, Inertia:{}".format(n_clusters_,
                                               inertia_dict[n_clusters_]))
    return inertia_dict, labels


def predict_kmeans(clusters_, df_main, df_normed):
    if type(clusters_) == int:
        clusters_ = [clusters_]
    for n_clusters_ in clusters_:
        model = get_kmeans(n_clusters_=n_clusters_)
        model = train(df_normed, model)
        df_main['cluster_KM_' + str(n_clusters_)] = model.predict(df_normed)
        print(df_main['cluster_KM_' + str(n_clusters_)].value_counts())
    return df_main, model


def get_dbscan(metric_p=1, eps_=0.5, min_samples=8, library='cuml'):
    if library == 'cuml':
        dbscan = cuml.DBSCAN(
            eps=eps_, min_samples=min_samples,
            metric='euclidean',
            verbose=0)
    else:
        dbscan = skcluster.DBSCAN(
            eps=eps_, min_samples=min_samples,
            p=metric_p,
            algorithm='auto',
            leaf_size=30)
    return dbscan


def rename_labels(ser_):
    csize = ser_.value_counts()
    csize = csize.sort_values(ascending=False)
    curr_labels = csize.index.to_arrow().to_pylist()
    new_labels = list(range(len(csize)))
    if -1 in curr_labels:
        neg1_idx = curr_labels.index(-1)
        new_labels[neg1_idx] = -1
    label_dict = dict(zip(curr_labels, new_labels))
    renamed_ser = cudf.Series([int(label_dict[x]) for x in ser_.to_arrow().to_pylist()])
    return renamed_ser


def iterate_dbscan(df_, metric_p=1, verbose=False, library='sklearn'):
    df_dbsc = df_.copy()
    labels = cudf.DataFrame()
    clust_size = dict()
    for eps_ in [0.0005*x for x in [1, 10, 20, 40, 100, 500, 1000, 2000, 3000, 5000, 10000]]:
        dbscan = get_dbscan(metric_p=metric_p, eps_=eps_, library=library)
        if library == 'sklearn':
            df_dbsc['cluster_dbscan'] = dbscan.fit_predict(df_dbsc.to_numpy())
        elif library == 'cuml':
            df_dbsc['cluster_dbscan'] = dbscan.fit_predict(df_dbsc)
        cluster_preds = df_dbsc['cluster_dbscan']
        if verbose:
            print("eps_value:{:.4f}, Found {} clusters".format(eps_, cluster_preds.nunique()))
        cluster_preds = rename_labels(cluster_preds)
        clust_size[eps_] = cluster_preds.value_counts()
        labels['eps_' + str(eps_)] = cluster_preds

    return clust_size, labels


def predict_dbscan(df_main, df_normed, eps_, metric_p=1):
    dbscan = get_dbscan(metric_p=metric_p, eps_=eps_)
    colname = 'cluster_dbscan_eps{:.4f}_minkp{}'.format(eps_,metric_p)
    df_main[colname] = dbscan.fit_predict(df_normed.values)

    return df_main, dbscan


def final_pass_kmeans(n_clusters_, df_main, df_normed, clusters_touse=5):
    """
    e.g., clusters_touse=5
    Take the top  5 clusters (by cluster size) and keep only these 5 cluster
     centers. In the final pass, assign each point to one of these 5 clusters.
    """

    model = get_kmeans(n_clusters_=n_clusters_)
    model = train(df_normed, model)
    df_main['cluster_KM' + str(n_clusters_)] = model.predict(df_normed)

    cluster_sizes = df_main['cluster_KM' + str(n_clusters_)].value_counts()
    retain_clusters = cluster_sizes.iloc[:clusters_touse]
    retain_clusters = retain_clusters.index.values

    dists = model.transform(df_normed)
    dists = dists[:, retain_clusters]
    closest_centers = np.argmin(dists, axis=1)
    cluster_nums = retain_clusters[closest_centers]
    df_main['finalpas_cluster_KM' + str(n_clusters_)] = cluster_nums

    print(df_main['finalpas_cluster_KM' + str(n_clusters_)].value_counts())
    return df_main, model



def draw_tsne(df_, init='random'):
    tsne = skmani.TSNE(n_components=2, learning_rate=100, init=init)
    return tsne.fit_transform(df_)


def draw_tsne_cuml(df_, perplexity=25.0, learning_rate=100.0):
    tsne = cuml.TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate)
    return tsne.fit_transform(df_)


def experiment_clust_methods(df_, df_norm_, models_=['KMeans', 'DBScan']):
    if 'KMeans' in models_:
        clust_nums = [8, 10, 12, 16]
        inertia_dict = iterate_kmeans(df_norm_)
        df_ = predict_kmeans(clust_nums, df_, df_norm_)
        df_, KM_model = final_pass_kmeans(18, df_, df_norm_, clusters_touse=5)

    if 'DBScan' in models_:
        print("\nMinkowski Param 1/2")
        iterate_dbscan(df_norm_, metric_p=0.5, verbose=True)

        print("\nManhattan Distance-Minkowski Param 1")
        iterate_dbscan(df_norm_, metric_p=1, verbose=True)

        print("\nEuclidean Distance-Minkowski Param 2")
        iterate_dbscan(df_norm_, metric_p=2, verbose=True)
    return


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


def get_silhouette_scores(df_, labels_,metric='euclidean', verbose=True):
    sh_sc = {}
    for label in labels_.columns:
        sh_sc[label] = silhouette_score(df_, labels_[label], metric=metric)
        if verbose:
            print("For clustering {}, Silhouette Score is {:.3f}".format(
                label, sh_sc[label]))
    return sh_sc


def tsneplot_util(df_, tsne_cols, color_map, title, clust):
    tsne1, tsne2 = tsne_cols[0], tsne_cols[1]
    df_['color']='k'
    for k,v  in color_map.items():
        df_.loc[df_[clust]==k,'color']=v
    scatter = plt.scatter(tsne1, tsne2, c='color', data=df_)
    plt.xlabel('tSNE1')
    plt.ylabel('tSNE2')
    plt.title(title)

@click.command()
@click.option('--model', default='dbscan', help='Clustering method to use.'\
     ' Valid choices are \'kmeans\' or \'dbscan\'. Default is \'dbscan\'')
@click.option('--experiment', is_flag=True,
    help='Boolean flag. If provided, script experiments by iterating over values for '\
     'parameters of the respective clustering method. When not provided,'\
     'trains and saves the model.')
def run(**kwargs):
    model = kwargs['model']
    experiment = kwargs['experiment']
    PCA_expl_variance = 0.9
    eps_dbsc = 0.0005
    clusters_km = 16

    compute_cluster_chars = False

    global NUM_DAYS, norm_cols
    data_fname = '../datasets/host_agg_data_day-01_day-10.csv'
    NUM_DAYS = 10.0

    assert model in ['kmeans', 'dbscan']

    df, df_norm = normalize_host_data(data_fname)

    # Perform PCA and do dimensionality reduction
    pca = cuml.PCA().fit(df_norm)
    expl_vars = pca.explained_variance_ratio_.to_pandas().to_list()
    cum_sum_vars = [sum(expl_vars[:idx+1]) for idx in range(len(expl_vars))]
    pca_dims = [i for i,var in enumerate(cum_sum_vars) if var > PCA_expl_variance]
    pca_dims = pca_dims[0]

    pca_cols = ['pca_'+str(x) for x in range(pca_dims)]
    df_norm[pca_cols] = pca.transform(df_norm).iloc[:,:pca_dims]
    df_pca = df_norm[pca_cols].copy()

    # Experiment or Training for the given clustering method
    if model=='dbscan':
        if experiment:
            experiment_clust_methods(df, df_pca, models_=['DBScan'])
        else:
            fname = "../models/dbscan_eps{}.pkl".format(eps_dbsc)
            df, dbsc_model = predict_dbscan(df, df_pca,  eps_=eps_dbsc, metric_p=1)
            pickle.dump((dbsc_model, pca, pca_dims), open(fname, "wb"))
            clust_ = 'cluster_dbscan_eps{}_minkp1'.format(eps_dbsc)
    elif model=='kmeans':
        if experiment:
            experiment_clust_methods(df, df_pca, models_=['KMeans'])
        else:
            fname = "../models/kmeans_{}clusts.pkl".format(clusters_km)
            df, kmeans_model = predict_kmeans(clusters_km, df, df_pca)
            pickle.dump((kmeans_model, pca, pca_dims), open(fname, "wb"))
            clust_ = 'cluster_KM_{}'.format(clusters_km)

    if not experiment and compute_cluster_chars:
        cluster_chars = compute_chars(df, clust_, cluster_id=0, NUM_DAYS=NUM_DAYS)

    return


if __name__ == '__main__':

    dt = datetime.date.today()
    logger_fname = 'logs/modeling.log'.format(dt.strftime('%d%m%y'))
    print("Logging in {}".format(logger_fname))
    logging.basicConfig(level=logging.DEBUG, filename=logger_fname,
                        filemode='a', format='%(asctime)s: %(message)s',
                        datefmt='%m%d-%H%M')
    run()