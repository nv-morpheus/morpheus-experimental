import datetime
import logging
import numpy as np
import sklearn.cluster as skcluster
import cudf
import cuml
from cuml.metrics.cluster import silhouette_score
from sklearn.metrics import silhouette_score as sk_silhouette_score
import pickle
import click
from utils import compute_chars, normalize_host_data


def pca_util(df_norm, pca_expl_variance):
    """
    Perform PCA and do dimensionality reduction.

    Returns the reduced dimension DataFrame
    """
    pca = cuml.PCA().fit(df_norm)
    expl_vars = pca.explained_variance_ratio_.to_pandas().to_list()
    cum_sum_vars = [sum(expl_vars[:idx+1]) for idx in range(len(expl_vars))]
    pca_dims = [i for i,var in enumerate(cum_sum_vars) if var > pca_expl_variance]
    pca_dims = pca_dims[0]

    pca_cols = ['pca_'+str(x) for x in range(pca_dims)]
    df_norm[pca_cols] = pca.transform(df_norm).iloc[:,:pca_dims]
    return df_norm, pca, pca_dims


def train(df_, model):
    """Fit the model with given DataFrame using .fit()"""
    model.fit(df_)
    return model


def get_kmeans(n_clusters=5):
    r"""Initialize and returns a KMeans model with n_clusters as num. of clusters"""
    kmeans_model = cuml.KMeans(n_clusters=n_clusters,
                               init='scalable-k-means++',
                               n_init=3, max_iter=300,
                               tol=0.0001, verbose=0)
    return kmeans_model


def iterate_kmeans(df, verbose=True, clust_min=2,clust_max=30, delta=2):
    """For KMeans, iterate over cluster sizes  - starting with clust_min, up to
    clust_max, in increments of delta.
    If verbose is True, outputs (num. of Clusters: inertia) to stdout

    Returns
    inertia_dict(dict): dictionary with (k,v)=(num. of clusters, inertia)

    labels (DataFrame): Rows: Hosts and Columns: Iterated "num of clusters"
    as parameter to KMeans
    """
    labels = cudf.DataFrame()
    inertia_dict = {}
    for n_clusters in range(clust_min, clust_max, delta):
        model = get_kmeans(n_clusters=n_clusters)
        model = train(df, model)
        inertia_dict[n_clusters] = model.inertia_
        labels['KMeans_'+str(n_clusters)] = model.predict(df)
        if verbose:
            print("Clusters:{}, Inertia:{}".format(n_clusters,
                                               inertia_dict[n_clusters]))
    return inertia_dict, labels


def predict_kmeans(clusters, df_main, df_normed):
    """
    Given num. of clusters in clusters (int or list of ints), returns
    df_main with clustering performed with each num.of clusters as input
    param for KMeans
    """
    if type(clusters) == int:
        clusters = [clusters]
    for n_clusters_ in clusters:
        model = get_kmeans(n_clusters_=n_clusters_)
        model = train(df_normed, model)
        df_main['cluster_KM_' + str(n_clusters_)] = model.predict(df_normed)
        print(df_main['cluster_KM_' + str(n_clusters_)].value_counts())
    return df_main, model


def final_pass_kmeans(n_clusters_, df_main, df_normed, clusters_touse=5):
    """
    Perform clustering using KMeans with n_clusters_ as number of clusters.

    One pass to keep only top 'clusters_touse' clusters and assign each
    data point to these 'clusters_touse' clusters.

    For e.g., clusters_touse=5, take the top  5 clusters (by size) and keep
    only these 5 cluster centers. In the final pass, assign each point to one
    of these 5 clusters.
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


def get_dbscan(metric_p=1, eps=0.5, min_samples=8, library='cuml'):
    """Initialize and returns a DBScan model with params eps, min_samples.

    library parameter determines whether a DBSCAN model from cuml or sklearn
    libraries is returned.

    If library=sklearn, metric_p is Minkowski ditance parameter.
    In case of cuml, Euclidean is the default distance metric.
    """

    assert library in ('cuml', 'sklearn'), "Valid choices are cuml or sklearn"
    if library == 'cuml':
        dbscan = cuml.DBSCAN(
            eps=eps, min_samples=min_samples,
            metric='euclidean',
            verbose=0)
    else:
        dbscan = skcluster.DBSCAN(
            eps=eps, min_samples=min_samples,
            p=metric_p,
            algorithm='auto',
            leaf_size=30)
    return dbscan


def iterate_dbscan(df, metric_p=1, verbose=True, library='sklearn'):
    """For DBSCAN, iterate over eps values and analyze number of clusters found
    for each eps value. This is the most important DBSCAN parameter to choose
    appropriately.

    library parameter determines whether a DBSCAN model from cuml or sklearn
    libraries is used. Valid choices are \'cuml\' or \'sklearn\'

    If library is 'sklearn', metric_p is Minkowski ditance parameter. In case of
    cuml, Euclidean is the default distance metric.

    If verbose is True, outputs eps: num. Clusters found to stdout

    Returns
    clust_size (dict): dictionary with (epsilon value, num. of clusters foundinertia)

    labels (DataFrame): Rows: Hosts and Columns: Clusters found for iterated
        eps values
    """
    # Iterates eps over a range of values from 5e-4 to 5. The range over which
    # to iterate depends entirely on the dataset and the range of distances
    # between different points in the dataset. Here we iterate over a large range.
    eps_iter = [0.0005*x for x in [1, 10, 20, 40, 100, 500, 1000, 2000, 3000, 5000, 10000]]

    df_dbsc = df.copy()
    labels = cudf.DataFrame()
    clust_size = dict()
    for eps in eps_iter:
        dbscan = get_dbscan(metric_p=metric_p, eps=eps, library=library)
        if library == 'sklearn':
            df_dbsc['cluster_dbscan'] = dbscan.fit_predict(df_dbsc.to_numpy())
        elif library == 'cuml':
            df_dbsc['cluster_dbscan'] = dbscan.fit_predict(df_dbsc)
        cluster_preds = df_dbsc['cluster_dbscan']
        if verbose:
            print("eps_value:{:.4f}, Found {} clusters".format(eps, cluster_preds.nunique()))
        cluster_preds = rename_labels(cluster_preds)
        clust_size[eps] = cluster_preds.value_counts()
        labels['eps_' + str(eps)] = cluster_preds

    return clust_size, labels


def predict_dbscan(df_main, df_normed, eps, metric_p=1):
    """
    Given the parameters \'eps\' and \'metric_p\', clustering is performed on
    \'df_normed'\ and a cluster value assigned to each datapoint

    Returns:
    df_main: DataFrame with an additional column representing DBSCAN clusters
    dbscan: Trained Clustering DBSCAN model
    """
    dbscan = get_dbscan(metric_p=metric_p, eps=eps)
    colname = 'cluster_dbscan_eps{:.4f}_minkp{}'.format(eps,metric_p)
    df_main[colname] = dbscan.fit_predict(df_normed.values)

    return df_main, dbscan


def rename_labels(ser):
    """
    Given a CuDf Series \'ser\' as input, where the index is hosts and values
    represent the cluster labels.

    rename_labels() renames the cluster labels to 0, 1,...,n in order of
    decreasing sizewhere n represents total number of clusters present
    excluding -1. Returns the Series with the renamed labels assigned to the
    hosts.
    """
    csize = ser.value_counts()
    csize = csize.sort_values(ascending=False)
    curr_labels = csize.index.to_arrow().to_pylist()
    new_labels = list(range(len(csize)))
    if -1 in curr_labels:
        neg1_idx = curr_labels.index(-1)
        new_labels[neg1_idx] = -1
    label_dict = dict(zip(curr_labels, new_labels))
    renamed_ser = cudf.Series([int(label_dict[x]) for x in ser.to_arrow().to_pylist()])
    return renamed_ser


def experiment_clust_methods(df_norm,
                            models=['KMeans', 'DBScan'],
                            km_clust_min=2,
                            km_clust_max=30,
                            km_clust_delta=2):
    """
    Iterate over relevant parameters for clustering methods in models parameter.

    If KMeans is in models, experiment for KMeans algorithm by iterating over
    cluster sizes - starting with clust_min, up to clust_max, in increments of delta.
    Outputs (num. of Clusters: inertia) to stdout

    If DBScan is in models, experiment for DBSCAN algorithm by iterating over
    distance metric and eps value, analyzing number of clusters found for each
    combination of parameter.
    Outputs, for each metric choice, eps: num. Clusters found to stdout
    """

    if 'KMeans' in models:
        _ = iterate_kmeans(df_norm,
                        clust_min=km_clust_min,
                        clust_max=km_clust_max,
                        delta=km_clust_delta)

    if 'DBScan' in models:
        print("Iterating for DBScan method using distance metrics:Minkowski Param= 1/2, 1, 2")
        print("\nMinkowski Param 1/2")
        iterate_dbscan(df_norm, metric_p=0.5)

        print("\nManhattan Distance-Minkowski Param 1")
        iterate_dbscan(df_norm, metric_p=1, verbose=True)

        print("\nEuclidean Distance-Minkowski Param 2")
        iterate_dbscan(df_norm, metric_p=2, verbose=True)


def get_silhouette_scores(df, labels, metric='euclidean', verbose=True, library='sklearn'):
    """Given a DataFrame with shape (hosts, features), and clustering labels
    for different clustering methods given by their column names in labels,
    returns average Silhouette score of overall datset for each clustering
    label/column
    """
    assert library in ('cuml', 'sklearn'), "Valid choices are cuml or sklearn"
    sh_sc = {}
    for label in labels.columns:
        if library == 'cuml':
            sh_sc[label] = silhouette_score(df, labels[label], metric=metric)
        else:
            dfpd = df.to_pandas()
            labelspd = labels.to_pandas()
            sh_sc[label] = sk_silhouette_score(dfpd, labelspd[label], metric=metric)
        if verbose:
            print("For clustering {}, Silhouette Score is {:.3f}".format(
                label, sh_sc[label]))
    return sh_sc


@click.command()
@click.option('--data_fname', default='host_agg_data_day-01_day-10.csv',\
     help='Name of the Preprocessed csv dataset to perofrm inference. The given'\
    'file name will be read from the relative path \'../datasets/ \'')
@click.option('--num_days', default=10.0, help='Number of days worth of data used'\
    'in preparing the dataset. Used to normalize the features.')
@click.option('--model', default='dbscan', help='Clustering method to use.'\
     ' Valid choices are \'kmeans\' or \'dbscan\'. Default is \'dbscan\'.'\
     'The corresponding model pickle file will be read from the relative'\
     'path \'../models/ \'.')
@click.option('--experiment', is_flag=True,
    help='Boolean flag. If provided, script experiments by iterating over values for '\
     'parameters of the respective clustering method. When not provided,'\
     'trains and saves the model.')
@click.option('--compute_cluster_chars', is_flag=True, help='Boolean flag. If '\
    'not provided, script just performs inference and output the cluster sizes.'\
    'If provided, additionally analyzes for the top salient features of each cluster'\
    'and prints the analysis to stdout.')
def run(**kwargs):
    dataset_path = '../datasets/'
    model_path = '../models/'
    pca_expl_variance = 0.9
    eps_dbsc = 0.0005
    clusters_km = 16

    data_fname = kwargs['data_fname']
    num_days = kwargs['num_days']
    model = kwargs['model']
    experiment = kwargs['experiment']
    compute_cluster_chars = kwargs['compute_cluster_chars']

    assert model in ['kmeans', 'dbscan'], "Valid choices for model are kmeans or dbscan"

    data_path =  dataset_path + data_fname
    df, df_norm = normalize_host_data(data_path)

    df_norm, pca, pca_dims = pca_util(df_norm, pca_expl_variance)
    pca_cols = ['pca_'+str(x) for x in range(pca_dims)]
    df_pca = df_norm[pca_cols].copy()

    # Experiment or Training for the given clustering method
    if model=='dbscan':
        if experiment:
            experiment_clust_methods(df, df_pca, models=['DBScan'])
        else:
            fname = model_path + "dbscan_eps{}.pkl".format(eps_dbsc)
            df, dbsc_model = predict_dbscan(df, df_pca,  eps=eps_dbsc, metric_p=1)
            pickle.dump((dbsc_model, pca, pca_dims), open(fname, "wb"))
            clust_ = 'cluster_dbscan_eps{}_minkp1'.format(eps_dbsc)
    elif model=='kmeans':
        if experiment:
            experiment_clust_methods(df, df_pca, models=['KMeans'])
        else:
            fname = model_path + "kmeans_{}clusts.pkl".format(clusters_km)
            df, kmeans_model = predict_kmeans(clusters_km, df, df_pca)
            pickle.dump((kmeans_model, pca, pca_dims), open(fname, "wb"))
            clust_ = 'cluster_KM_{}'.format(clusters_km)

    if not experiment and compute_cluster_chars:
        cluster_chars = compute_chars(df, clust_, cluster_id=0, num_days=num_days)

    return


if __name__ == '__main__':

    dt = datetime.date.today()
    logger_fname = 'logs/modeling.log'.format(dt.strftime('%d%m%y'))
    print("Logging in {}".format(logger_fname))
    logging.basicConfig(level=logging.DEBUG, filename=logger_fname,
                        filemode='a', format='%(asctime)s: %(message)s',
                        datefmt='%m%d-%H%M')
    run()
