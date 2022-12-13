import datetime
import logging
import pickle
import click
from utils import compute_chars
from train import normalize_host_data


@click.command()
@click.option('--model', default='dbscan', help='Clustering method to use.'\
     ' Valid choices are \'kmeans\' or \'dbscan\'. Default is \'dbscan\'')
@click.option('--data_fname', default='host_agg_data_day-11_day-15.csv',\
     help='Name of the Preprocessed csv dataset to perofrm inference.')
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

    data_path =  dataset_path + kwargs['data_fname']
    df, df_norm = normalize_host_data(data_path)

    if model=='dbscan':
        fname = model_path + 'dbscan_eps0.0005.pkl'
        clust_ = "cluster_dbscan_eps0.0005_minkp1"

        dbsc_model, pca, pca_dims = pickle.load(open(fname, "rb"))
        df_pca = pca.transform(df_norm).iloc[:,:pca_dims]
        df[clust_] = dbsc_model.fit_predict(df_pca)

    elif model=='kmeans':
        fname = model_path + 'kmeans_16clusts.pkl'
        clust_ = "cluster_KM_16"

        kmeans_model, pca, pca_dims = pickle.load(open(fname, "rb"))
        df_pca = pca.transform(df_norm).iloc[:,:pca_dims]
        df[clust_] = kmeans_model.predict(df_pca)

    print("Cluster Size:\n{}".format(df[clust_].value_counts()))

    if compute_cluster_chars:
        cluster_chars = compute_chars(df, clust_, cluster_id=0, num_days=num_days)

    return


if __name__ == '__main__':
    dt = datetime.date.today()
    logger_fname = 'logs/inference.log'.format(dt.strftime('%d%m%y'))
    print("Logging in {}".format(logger_fname))
    logging.basicConfig(level=logging.DEBUG, filename=logger_fname,
                        filemode='a', format='%(asctime)s: %(message)s',
                        datefmt='%m%d-%H%M')
    run()
