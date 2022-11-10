import cudf
import bz2
import logging
import time
import click
import numpy as np
from utils import *
from collections import defaultdict
#get_fnames, read_wls, compute_logins_with_loghostuname, compute_eventid_cnt, compute_diff_source_logon_cnt, compute_eventid_cnt_source, logon_types, account_logons, compute_username_domain_cnt, compute_username_cnt

def host_aggr(df_, host_, dict_, count_cols):
    newhosts = set(df_['LogHost'].to_pandas()).union(set(df_['Source'].to_pandas()))
    newhosts = newhosts - set(host_.index.to_pandas())

    frac_cols = ['uname_other_compacnt_login_frac','uname_that_compacnt_login_frac']
    newhost = cudf.DataFrame({'LogHost': newhosts}).set_index('LogHost')
    newhost[count_cols] = 0
    newhost[frac_cols] = 0.0

    if host_.shape[0] == 0:
        host_ = newhost.copy()
    else:
        host_ = cudf.concat([host_, newhost], axis=0)
    numrows = df_.shape[0]
    ##Remove rows if Both SOURCE & DESTINATION neq NA
    df_ = df_.loc[(df_['Source'].isna()) | (df_['Destination'].isna())]
    if numrows < df_.shape[0]:
        logging.debug("Filtering Rows if SOURCE & DESTINATION neq NA")
        logging.debug("Removed {} ROWS".format(numrows-df_.shape[0]))

    host_ = compute_logins_with_loghostuname(df_, host_)
    host_ = logon_types(df_, host_, valid_logon_types)
    host_, dict_ = compute_diff_source_logon_cntv1(df_, host_, dict_)
    host_, dict_ = compute_username_cntv1(df_, host_, dict_)
    host_, dict_ = compute_username_domain_cntv1(df_, host_, dict_)

    for evtuple in evtuples:
        evid, ev_str = evtuple
        host_ = compute_eventid_cnt(df_ , evid, ev_str, host_)

    for evtuple in evtuples_src:
        evid, ev_str = evtuple
        host_ = compute_eventid_cnt_source(df_ , evid, ev_str, host_)
    host_[count_cols] = host_[count_cols].fillna(value=0, inplace=False)
    host_['uname_other_compacnt_login_frac'] = host_['uname_other_compacnt_login_cnt']/host_['total_logins_cnt']
    host_['uname_other_compacnt_login_frac'] = host_['uname_other_compacnt_login_frac'].replace(np.inf, -1.)

    host_['uname_that_compacnt_login_frac'] = host_['uname_that_compacnt_login_cnt']/host_['total_logins_cnt']
    host_['uname_that_compacnt_login_frac'] = host_['uname_that_compacnt_login_frac'].replace(np.inf, -1.)

    return host_, dict_


def initialize_hostdf():
    global valid_logon_types, evtuples, evtuples_src
    valid_logon_types = {0, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12}
    evtuples = [
        (4624, 'total_logins_cnt'),
        (4625, 'accnt_fail_logon_cnt'),
        (4634, 'total_logoff_cnt'),
        (4647, 'total_user_initi_logoff_cnt'),
        (4648, 'logon_explicit_cred_frm_cnt'),
        (4672, 'spl_pvlgs'),
        (4776, 'domain_ctr_validate_cnt'),
        (4802, 'scrnsaver_invok_cnt'),
        (4803, 'scrnsaver_dismiss_cnt')]
    #(4769, 'TGS_req_cnt'),(4768, 'TGT_req_cnt') removed since 100% of LogHost for 4768,4769 is ActiveDirectory
    evtuples_dst = [ (4648, 'logon_explicit_cred_to_cnt')]
    evtuples_src = [
        (4624, 'total_logins_src_cnt'),
        (4625, 'accnt_fail_logon_src_cnt'),
        (4768, 'TGT_req_src_cnt'),
        (4769, 'TGS_req_src_cnt'),
        (4776, 'domain_ctr_validate_src_cnt')
        ]

    count_cols = ['UserName_cnt', 'DomainName_cnt', 'Source_cnt']
    count_cols += [x[1] for x in evtuples+evtuples_src+evtuples_dst]
    count_cols += ['logon_type_{}'.format(int(x)) for x in valid_logon_types]
    count_cols += ['logon_type_frm_{}'.format(int(x)) for x in valid_logon_types]

    count_cols += ['uname_other_compacnt_login_cnt', 'uname_that_compacnt_login_cnt']
    host = cudf.DataFrame(columns=['LogHost']).set_index('LogHost')

    dict_ = {'Sources':defaultdict(set),
             'Unames':defaultdict(set),
             'UserDomains':defaultdict(set)
             }
    return host, dict_, count_cols


def read_process_data(wls_files):
    host_, dict_, count_cols = initialize_hostdf()
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
            host_, dict_ = host_aggr(df_wls, host_, dict_, count_cols)

            total_lines += len(current_block)/1000000
            iter_ += 1

            if iter_ % 1 == 0:
                logging.info('{:.3f}M Lines, {:.2f}K/sec'.format(
                    total_lines, 1000.0*total_lines / (time.time() - t0)))
                logging.debug('host shape:{}'.format(host_.shape))
            if total_lines*1e6 > MAX_LINES:
                    logging.info("Breaking for loop. total_lines={}>{}".format(total_lines, MAX_LINES))
                    break
        fi.close()
    return host_


@click.command()
@click.option('--debug', is_flag=True)
def run(**kwargs):
    global dataset_path, readsize, MAX_LINES
    debug_mode = kwargs['debug']
    logging.basicConfig(level=logging.DEBUG, datefmt='%m%d-%H%M',
                        format='%(asctime)s: %(message)s')
    dataset_path = '../data/'
    if debug_mode:
        ipfile_suffix = 'day-01_day-01'
        MAX_LINES = 5e6
        readsize = 32768*32
        opfile_suffix = '_{:d}Mlines'.format(int(MAX_LINES / 1e6))
    else:
        ipfile_suffix = 'day-01_day-10'
        MAX_LINES = 1e12
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
    host_ = read_process_data(wls_files)
    logging.debug("Number of hosts:{}".format(host_.shape[0]))
    host_.to_csv(dataset_path + 'aggr/host_agg_data{}.csv'.format(opfile_suffix))


if __name__ == '__main__':
   run()
