import cudf
import bz2
import logging
import time
import click
import numpy as np
from utils import *
from collections import defaultdict
from itertools import chain

VALID_LOGON_TYPES = {0, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12}

# List of tuples (EventID, feature name), where a feature name denotes
# frequency of corresp. EventID, by asset appearing in LogHost field.
EVENTID_CNTFEAT = [
    (4624, 'total_logins_cnt'),
    (4625, 'accnt_fail_logon_cnt'),
    (4634, 'total_logoff_cnt'),
    (4647, 'total_user_initi_logoff_cnt'),
    (4648, 'logon_explicit_cred_frm_cnt'),
    (4672, 'spl_pvlgs'),
    (4776, 'domain_ctr_validate_cnt'),
    (4802, 'scrnsaver_invok_cnt'),
    (4803, 'scrnsaver_dismiss_cnt')]
# (4768, 'TGT_req_cnt'), (4769, 'TGS_req_cnt')
# 4768 & 4769 not used since 100% of LogHost for 4768,4769 is ActiveDirectory

# EVENTIDFORSOURCE_CNTFEAT & EVENTIDFORDEST_CNTFEAT are similar to EVENTID_CNTFEAT
# except that they corresp. to frequency of an EventID, by asset, appearing in
#  Source & Destination fields resp.
EVENTIDFORSOURCE_CNTFEAT = [
    (4624, 'total_logins_src_cnt'),
    (4625, 'accnt_fail_logon_src_cnt'),
    (4768, 'TGT_req_src_cnt'),
    (4769, 'TGS_req_src_cnt'),
    (4776, 'domain_ctr_validate_src_cnt')
    ]
EVENTIDFORDEST_CNTFEAT = [(4648, 'logon_explicit_cred_to_cnt')]


def host_aggr(df_, host_, uniq_values_dict, count_cols):
    """
    Args:
        df_: cudf DataFrame with the data read from windows event logs file

        host_: DataFrame of hosts seen so far, with aggregated features

        uniq_values_dict: Dictionary with (k,v) pairs being (field, dict_)
            where dict_ represents, for the given 'field', a dictionary of (k,v)
            pairs of (host, Set of unique values in 'field' for that host)

        count_cols: List of features that represents counts

    Returns:
        host_: Updated host_ with data from df_

        uniq_values_dict: Updated uniq_values_dict with data from df_
    """

    newhosts = set(df_['LogHost'].to_pandas()).union(set(df_['Source'].to_pandas()))
    newhosts = newhosts - set(host_.index.to_pandas())
    newhosts.discard(None)

    frac_cols = ['uname_other_compacnt_login_frac','uname_that_compacnt_login_frac']
    newhost = cudf.DataFrame({'LogHost': newhosts}).set_index('LogHost')
    newhost[count_cols] = 0
    newhost[frac_cols] = 0.0

    if host_.shape[0] == 0:
        host_ = newhost.copy()
    else:
        host_ = cudf.concat([host_, newhost], axis=0)

    numrows = df_.shape[0]
    # Remove rows if Both SOURCE & DESTINATION neq NA since LogHost is meaningless
    # in a directed authentication event with both SOURCE & DESTINATION present.
    df_ = df_.loc[(df_['Source'].isna()) | (df_['Destination'].isna())]
    if numrows < df_.shape[0]:
        logging.debug("Filtering Rows if SOURCE & DESTINATION neq NA")
        logging.debug("Removed {} ROWS".format(numrows-df_.shape[0]))

    host_ = compute_logins_with_loghostuname(df_, host_)
    host_ = logon_types(df_, host_, VALID_LOGON_TYPES)
    host_, uniq_values_dict = compute_diff_source_logon_cnt(df_, host_, uniq_values_dict)
    host_, uniq_values_dict = compute_username_cnt(df_, host_, uniq_values_dict)
    host_, uniq_values_dict = compute_username_domain_cnt(df_, host_, uniq_values_dict)

    for evtuple in EVENTID_CNTFEAT:
        evid, ev_str = evtuple
        host_ = compute_eventid_cnt(df_ , evid, ev_str, host_)

    for evtuple in EVENTIDFORSOURCE_CNTFEAT:
        evid, ev_str = evtuple
        host_ = compute_eventid_cnt_source(df_ , evid, ev_str, host_)
    host_[count_cols] = host_[count_cols].fillna(value=0, inplace=False)
    host_['uname_other_compacnt_login_frac'] = host_['uname_other_compacnt_login_cnt']/host_['total_logins_cnt']
    host_['uname_other_compacnt_login_frac'] = host_['uname_other_compacnt_login_frac'].replace(np.inf, -1.)

    host_['uname_that_compacnt_login_frac'] = host_['uname_that_compacnt_login_cnt']/host_['total_logins_cnt']
    host_['uname_that_compacnt_login_frac'] = host_['uname_that_compacnt_login_frac'].replace(np.inf, -1.)

    return host_, uniq_values_dict


def initialize_hostdf():
    """
    Initializes and returns following variables.

    Returns:

        host: empty cudf DataFrame representing assets/hosts

        uniq_values_dict: dictionary with fields for which unique values
            encountered need to be tracked

        count_cols: features that are counts; computed by counting number
            of occurrences
    """

    count_cols = ['UserName_cnt', 'DomainName_cnt', 'Source_cnt']
    count_cols += [x[1] for x in chain(EVENTID_CNTFEAT, EVENTIDFORSOURCE_CNTFEAT, EVENTIDFORDEST_CNTFEAT)]
    count_cols += ['logon_type_{}'.format(int(x)) for x in VALID_LOGON_TYPES]
    count_cols += ['logon_type_frm_{}'.format(int(x)) for x in VALID_LOGON_TYPES]

    count_cols += ['uname_other_compacnt_login_cnt', 'uname_that_compacnt_login_cnt']
    host = cudf.DataFrame(columns=['LogHost']).set_index('LogHost')

    uniq_values_dict = {
        'Sources': defaultdict(set),
        'Unames': defaultdict(set),
        'UserDomains': defaultdict(set)
        }
    return host, uniq_values_dict, count_cols


def read_process_data(wls_files, readsize=1000000, max_lines=1e15):
    """
    Reads each file from input list, iteratively block by block, does feature
    computation and aggregates derived features by host.

    Args:
        wls_files (list): List of windows event log files, compressed using bzip2

        readsize (int): Number of Bytes of data in each iteration

        max_lines (int): Maximum number of lines to read before breaking

    Returns:
        DataFrame with shape (number of hosts, number of derived features)

    """

    host_df, uniq_vals_dict, count_cols = initialize_hostdf()
    for wls_fname in wls_files:
        residue = b''
        decomp = bz2.BZ2Decompressor()
        total_lines, iter_, t0 = 0, 0, time.time()

        fi = open(wls_fname, 'rb')
        for data in iter(lambda: fi.read(readsize), b''):
            raw = residue + decomp.decompress(data)
            current_block = raw.split(b'\n')
            residue = current_block.pop()  # last line could be incomplete
            df_wls = read_wls(current_block, file_path=False)
            host_df, uniq_vals_dict = host_aggr(df_wls, host_df, uniq_vals_dict, count_cols)

            total_lines += len(current_block)/1000000
            iter_ += 1

            if iter_ % 10000 == 0:
                proc_speed = 1000.0*total_lines / (time.time() - t0)
                logging.info(
                    '{:.3f}M Lines, {:.2f}K/sec'.format(total_lines, proc_speed))
                logging.debug('host shape:{}'.format(host_df.shape))
            if total_lines*1e6 > max_lines:
                    logging.info("Breaking for loop. total_lines={}>{}".format(total_lines, max_lines))
                    break
        fi.close()
    return host_df


@click.command()
@click.option('--debug', is_flag=True)
@click.option('--data_range', default='day-01-day-01',
     help='Range of dates for which wls files need to be read and preprocessed. '\
     'For example, data_range=day-01-day_03 reads wls_day-01.bz2, wls_day-02.bz2'\
     'and wls_day-03.bz2, preprocess them and prepare a combined dataset.')
def run(**kwargs):
    global dataset_path
    debug_mode = kwargs['debug']
    logging.basicConfig(level=logging.DEBUG, datefmt='%m%d-%H%M',
                        format='%(asctime)s: %(message)s')
    dataset_path = '../datasets/'
    ipfile_suffix = kwargs['data_range']
    if debug_mode:
        max_lines = 5e6
        readsize = 32768*32
        opfile_suffix = '_{:d}Mlines'.format(int(max_lines / 1e6))
    else:
        max_lines = 1e15
        readsize = 32768*32*30
        opfile_suffix = '_' + ipfile_suffix
        logger_fname = 'logs/dataprocess_{}.log'.format(ipfile_suffix)
        fh = logging.FileHandler(filename=logger_fname, mode='a')
        fmt = logging.Formatter('%(asctime)s: %(message)s', datefmt='%m%d-%H%M')
        fh.setFormatter(fmt)
        logging.getLogger().addHandler(fh)
        print("Logging in {}".format(logger_fname))

    logging.info("DataProcess for WLS files {}. Read Size:{}MB\n\n".format(
        ipfile_suffix, readsize//2**20))
    wls_files = get_fnames(dataset_path, ipfile_suffix)
    host_df = read_process_data(wls_files, readsize, max_lines)
    logging.debug("Number of hosts:{}".format(host_df.shape[0]))
    host_df.to_csv(dataset_path + 'aggr/host_agg_data{}.csv'.format(opfile_suffix))


if __name__ == '__main__':

   run()
