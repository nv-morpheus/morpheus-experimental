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

import logging
import os

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.manifold as skmani

import cudf
import cuml
import cuml.preprocessing as cupreproc


def read_netflow(fname, nrows=None):
    """
    Given full path of a netflow csv file at fname, reads the data.
    Assumes the file has columns compatible with LANL netflow dataset.
    """
    df = cudf.read_csv(fname, nrows=nrows)
    netflow_header = [
        'time',
        'Duration',
        'SrcDevice',
        'DstDevice',
        'Protocol',
        'SrcPort',
        'DstPort',
        'SrcPackets',
        'DstPackets',
        'SrcBytes',
        'DstBytes'
    ]
    df.columns = netflow_header
    df['time_h'] = cudf.to_datetime(df['time'], unit='ms')
    return df


def read_wls(file_to_read, file_path=False, nrows=None):
    """
    Read the windows event log file.
    If file_path is True, file_to_read is assumed to be full path
    If file_path is False, file_to_read is assumed to be json text

    Returns cudf DataFrame.
    """
    if file_path:
        df = cudf.read_json(file_to_read, lines=True, nrows=nrows)
    else:
        txt = "\n".join([x.decode("utf-8") for x in file_to_read])
        df = cudf.read_json(txt, lines=True)

    df['time_dt'] = cudf.to_datetime(df['Time'], unit='s')  # format='%Y-%m-%d %H:%M:%S.%f')
    return df


def normalize_host_data(data_fname, preproc='minmax', norm_method='l2'):
    """
    Reads the preprocessed dataset and normalizes the individual features.

    Args:
        data_fname (str): full path at which the preprocessed dataset is saved

        preproc (str): Valid choices are minmax and unit_norm

        norm_method (str): Vald choices are l1 or l2. Applicable only when \'preproc = unit_norm

    Returns:
        df (DataFrame): cudf DataFrame with non-normalized data

        df_norm (DataFrame): cudf DataFrame with normalized data
    """
    assert preproc in ('minmax', 'unit_norm'), "Valid choices are minmax or unit_norm"

    df = cudf.read_csv(data_fname)
    print("Num. of columns:{}".format(len(df.columns)))

    rm_assets = ['ActiveDirectory', 'EnterpriseAppServer']
    print("\nREMOVED {} Assets from data".format(rm_assets))
    df = df.loc[~df['LogHost'].isin(rm_assets)]

    rm_cols = ['LogHost', 'uname_other_compacnt_login_frac', 'uname_that_compacnt_login_frac']
    norm_cols = [x for x in df.columns if x not in rm_cols]

    if preproc == 'unit_norm':
        scaler = cupreproc.normalize(norm=norm_method)
    elif preproc == 'minmax':
        scaler = cupreproc.MinMaxScaler(feature_range=(0, 1))

    df_norm = scaler.fit(df[norm_cols]).transform(df[norm_cols])
    df_norm.columns = norm_cols

    return df, df_norm


def compute_username_cnt(df, host, srcdict):
    """
    From Windows Event Logs data, counts number of unique UserNames for each
    host and updates the count in host DataFrame and unique values in srcdict

    Args:
        df(DataFrame): Current block of Windows Event Logs data to process

        host(DataFrame): Values of features computed so far, for each host

        srcdict (dict): Dictionary with (k,v) pairs being (field, dict_)
            where dict_ represents, for any given 'field', a dictionary of
            (k,v) pairs of (host, Set of unique values in 'field' for that host)

    Returns:
        host(DataFrame):host with updated values in 'UserName_cnt' column

        srcdict (dict): srcdict with updated dictionary srcdict['Unames']
            where the uniques UserNames seen so far for each host is updated in
            the dict srcdict['Unames']
    """
    df = df[['LogHost', 'UserName']].copy()
    df = df.loc[~df['UserName'].isna()]

    unique_usernames = df.groupby('LogHost')['UserName'].agg('unique')
    unique_usernames = unique_usernames.rename('unique_usernames').to_pandas()

    for i in range(unique_usernames.shape[0]):
        hostval, unames = unique_usernames.index[i], unique_usernames.iloc[i]
        srcdict['Unames'][hostval] = srcdict['Unames'][hostval].union(unames)

    uname_cnt_df = cudf.DataFrame({
        'LogHost': srcdict['Unames'].keys(), 'UserName_cnt': [len(v) for v in srcdict['Unames'].values()]
    })

    comb = cudf.merge(host['UserName_cnt'].reset_index(), uname_cnt_df, how='outer', on='LogHost')

    # DomainName_cnt_x has DomaiName counts upto prev chunk, DomainName_cnt_y has
    # updated DomaiName counts only for hosts present in new data.
    comb.loc[~comb['UserName_cnt_y'].isna(), 'UserName_cnt_x'] = 0
    comb = comb.fillna({'UserName_cnt_y': 0})

    comb['UserName_cnt'] = comb['UserName_cnt_x'] + comb['UserName_cnt_y']
    comb = comb.drop(['UserName_cnt_x', 'UserName_cnt_y'], axis=1).set_index('LogHost')
    host = host.drop('UserName_cnt', axis=1)
    host = cudf.merge(host, comb, how='inner', on='LogHost')

    return host, srcdict


def compute_username_domain_cnt(df, host, srcdict):
    """
    From Windows Event Logs data, counts number of unique DomainName for each
    host and updates a. count in host DataFrame and b. unique values in srcdict

    Args:
        df(DataFrame): Current block of Windows Event Logs data to process

        host(DataFrame): Values of features computed so far, for each host

        srcdict (dict): Dictionary with (k,v) pairs being (field, dict_)
            where dict_ represents, for any given 'field', a dictionary of
            (k,v) pairs of (host, Set of unique values in 'field' for that host)

    Returns:
        host(DataFrame):host with updated values in 'DomainName_cnt' column

        srcdict (dict): srcdict with updated dictionary srcdict['UserDomains']
            where the uniques DomainNames seen so far for each host is updated in
            the dict srcdict['UserDomains']
    """

    df = df[['LogHost', 'DomainName']].copy()
    df = df.loc[~df['DomainName'].isna()]

    unique_username_domains = df.groupby('LogHost')['DomainName'].agg('unique')
    unique_username_domains = unique_username_domains.rename('unique_username_domains').to_pandas()

    for i in range(unique_username_domains.shape[0]):
        hostval, unames = unique_username_domains.index[i], unique_username_domains.iloc[i]
        srcdict['UserDomains'][hostval] = srcdict['UserDomains'][hostval].union(unames)

    udomain_cnt_df = cudf.DataFrame({
        'LogHost': srcdict['UserDomains'].keys(), 'DomainName_cnt': [len(v) for v in srcdict['UserDomains'].values()]
    })
    udomain_cnt_df = udomain_cnt_df.set_index('LogHost', drop=True)

    comb = cudf.merge(host['DomainName_cnt'].reset_index(), udomain_cnt_df, how='outer', on='LogHost')

    # DomainName_cnt_x has DomaiName counts upto prev chunk, DomainName_cnt_y has
    # updated DomaiName counts only for hosts present in new data.
    comb.loc[~comb['DomainName_cnt_y'].isna(), 'DomainName_cnt_x'] = 0
    comb = comb.fillna({'DomainName_cnt_y': 0})

    comb['DomainName_cnt'] = comb['DomainName_cnt_x'] + comb['DomainName_cnt_y']
    comb = comb.drop(['DomainName_cnt_x', 'DomainName_cnt_y'], axis=1).set_index('LogHost')

    host = host.drop('DomainName_cnt', axis=1)
    host = cudf.merge(host, comb, how='inner', on='LogHost')

    host = cudf.merge(host, udomain_cnt_df, how='outer', on='LogHost')
    host = host.drop(['DomainName_cnt_x'], axis=1)
    host = host.rename({'DomainName_cnt_y': 'DomainName_cnt'}, axis=1)
    return host, srcdict


def logon_types(df, host, valid_logon_types):
    """
    Computes number of logins by each LogonType present in valid_logon_types.

    Counts for each host present in LogHost and Source columns separately.

    Returns:
        host(DataFrame): host with updated values in 'logon_type_x' and
             'logon_type_frm_x' columns
    """

    def cnt_logontypes(df_touse, logon_types, host, suffix=''):
        for ltype in logon_types:
            col_name = 'logon_type_{}{}'.format(suffix, int(ltype))
            df_ltype = df_touse.loc[df_touse['LogonType'] == ltype]
            dfltype_cnt = df_ltype['LogHost'].value_counts().rename(col_name)
            dfltype_cnt.index.rename('LogHost', inplace=True)
            host = cudf.merge(host, dfltype_cnt, on='LogHost', how='left')
            host[col_name] = host[col_name + '_x'] + host[col_name + '_y']
            host.drop([col_name + '_x', col_name + '_y'], axis=1, inplace=True)
        return host

    df = df.loc[df['EventID'].isin([4624, 4625])]
    ltype_indf = df['LogonType'].unique()
    logon_types_ = ltype_indf.loc[ltype_indf.isin(valid_logon_types)].to_pandas().to_list()
    host = cnt_logontypes(df, logon_types_, host)

    df_src = df.loc[~df['Source'].isna()]
    ltype_indf = df_src['LogonType'].unique()
    logon_types_ = ltype_indf.loc[ltype_indf.isin(valid_logon_types)].to_pandas().to_list()
    host = cnt_logontypes(df_src, logon_types_, host, suffix='frm_')

    return host


def compute_diff_source_logon_cnt(df, host, srcdict):
    """
    For each LogHost, compute total number of unique hosts in 'Source'
    Does not filter by EventID, considers all EventTypes.
    """

    df = df[['LogHost', 'Source']].copy()
    df = df.loc[~df['Source'].isna()]

    unique_sources = df.groupby('LogHost')['Source'].agg('unique')
    unique_sources = unique_sources.rename('unique_sources').to_pandas()

    for i in range(unique_sources.shape[0]):
        hostval, srces = unique_sources.index[i], unique_sources.iloc[i]
        srcdict['Sources'][hostval] = srcdict['Sources'][hostval].union(srces)

    src_cnt_df = cudf.DataFrame({
        'LogHost': srcdict['Sources'].keys(), 'Source_cnt': [len(v) for v in srcdict['Sources'].values()]
    })

    comb = cudf.merge(host['Source_cnt'].reset_index(), src_cnt_df, how='outer', on='LogHost')

    # Source_cnt_x has source counts from prev, Source_cnt_y has updated source
    # counts only for hosts present in new data.
    comb.loc[~comb['Source_cnt_y'].isna(), 'Source_cnt_x'] = 0
    comb = comb.fillna({'Source_cnt_y': 0})

    comb['Source_cnt'] = comb['Source_cnt_x'] + comb['Source_cnt_y']
    comb = comb.drop(['Source_cnt_x', 'Source_cnt_y'], axis=1).set_index('LogHost')
    host = host.drop('Source_cnt', axis=1)
    host = cudf.merge(host, comb, how='inner', on='LogHost')

    return host, srcdict


def compute_logins_with_loghostuname(df, host, login_eventids=[4624, 4625]):
    """
    Computes logins from the username corresponding to
    a. computer accounts corresp. to specified LogHost i.e. UserName= LogHost+'$'
    b. computer accounts corresp. to other LogHost i.e. UserName ending with $ and != LogHost+'$'
    """
    df = df.loc[df['EventID'].isin(login_eventids)]
    df_1 = df.loc[(df['UserName'].str.endswith('$')) & (df['UserName'] != df['LogHost'] + '$')]

    uname_other_compacnt_login_cnt = df_1['LogHost'].value_counts()\
                                                 .rename('uname_other_compacnt_login_cnt')

    uname_other_compacnt_login_cnt.index.rename('LogHost', inplace=True)
    host = cudf.merge(host, uname_other_compacnt_login_cnt, how='outer', on='LogHost')

    df_2 = df.loc[df['UserName'] == df['LogHost'] + '$']
    uname_that_compacnt_login_cnt = df_2['LogHost'].value_counts()\
                                                 .rename('uname_that_compacnt_login_cnt')
    uname_that_compacnt_login_cnt.index.rename('LogHost', inplace=True)
    host = cudf.merge(host, uname_that_compacnt_login_cnt, how='outer', on='LogHost')

    for col in ['uname_other_compacnt_login_cnt', 'uname_that_compacnt_login_cnt']:
        host[col] = host[col + '_x'] + host[col + '_y']
        host.drop([col + '_x', col + '_y'], axis=1, inplace=True)
    return host


def compute_eventid_cnt(df, evid, ev_str, host):
    """
    For each asset=i, Counts the number of rows with
     EventID == evid &
     LogHost == i
    and updates as follows: host[i][ev_str] += count
    """
    df_evid = df.loc[df['EventID'] == evid]
    event_cnt = df_evid['LogHost'].value_counts().rename(ev_str)
    event_cnt.index.rename('LogHost', inplace=True)

    if set(event_cnt.index.to_pandas()) - set(host.index.to_pandas()):
        logging.error("Found extra LogHosts. UNEXPECTED BEHAVIOR")
    host = cudf.merge(host, event_cnt, how='left', on='LogHost')
    host[ev_str] = host[ev_str + '_x'] + host[ev_str + '_y']
    host.drop([ev_str + '_y', ev_str + '_x'], axis=1, inplace=True)

    return host


def compute_eventid_cnt_source(df, evid, ev_str, host):
    """
    For each asset=i, counts the number of rows with
     EventID == evid &
     Source == i
    and updates as follows: host[i][ev_str] += count
    """
    df_evid = df.loc[df['EventID'] == evid]
    event_cnt = df_evid['Source'].value_counts().rename(ev_str)
    event_cnt.index.rename('LogHost', inplace=True)

    if set(event_cnt.index.to_pandas()) - set(host.index.to_pandas()):
        logging.error("Found extra LogHosts. UNEXPECTED BEHAVIOR")
    host = cudf.merge(host, event_cnt, how='left', on='LogHost')
    host[ev_str] = host[ev_str + '_x'] + host[ev_str + '_y']
    host.drop([ev_str + '_y', ev_str + '_x'], axis=1, inplace=True)

    return host


def get_fnames(path, day_range):
    """
    Given the range of days in 'day_range', reads the files with format
    'wls_day-xx' at 'path' and returns the files that fall with in the
    ranage, inclusive of the days provided in 'day_range'.
    """

    start_day, end_day = day_range.split('_')
    wls_files = [x for x in os.listdir(path) if x.startswith('wls_day')]
    day_tags = [x.split('_')[1].split('.bz2')[0] for x in wls_files]

    wls_files = [wls_fname for idx, wls_fname in enumerate(wls_files) if start_day <= day_tags[idx] <= end_day]
    wls_files = sorted(wls_files)
    wls_files = [path + x for x in wls_files]
    return wls_files


def hist_util(df0, col, clust, num_bins=8):
    coldf = df0.dropna(axis=0, subset=[col])
    col_clust0, col_clustrest = coldf.loc[coldf[clust] == 0, col], coldf.loc[coldf[clust] != 0, col]

    val_25pct, val_75pct = np.percentile(col_clust0.loc[col_clust0 != 0], [25, 75])

    binw = (val_75pct - val_25pct) * 2 / num_bins
    bins = [val_25pct + i * binw for i in range(-num_bins // 4, 3 * num_bins // 4 + 1)]
    clust0_hist, clust0_vals = np.histogram(col_clust0, bins=num_bins)
    clustrem_hist, clustrem_vals = np.histogram(col_clustrest, bins=clust0_vals)

    clust0_hist, clustrem_hist = cp.asnumpy(clust0_hist), cp.asnumpy(clustrem_hist)

    return clust0_hist, clustrem_hist, bins


def compute_val_counts(df, col, clust):
    freq_0 = df.loc[df[clust] == 0][col].value_counts()
    freq_rem = df.loc[df[clust] != 0][col].value_counts()

    freq_0, freq_rem = 100 * freq_0 / freq_0.sum(), 100 * freq_rem / freq_rem.sum()
    freqs = pd.merge(freq_0, freq_rem, left_index=True, right_index=True, how='outer')
    freqs.fillna(0, inplace=True)
    return freqs


def compute_chars(df,
                  clust,
                  num_days=1,
                  cluster_id='all',
                  write_differences=False,
                  verbose=False,
                  top_diff_summary_feats=10,
                  top_diff_detail_feats=8):
    clusters = df[clust].value_counts().rename('clust_size')
    clusters = clusters.reset_index().rename({'index': clust}, axis=1)
    ignore_cols = [clust, 'LogHost', 'num_accnt_logons', 'num_accnt_succ_logons']

    groupby_clust = df.groupby(clust, as_index=False)
    for col in set(df.columns) - set(ignore_cols):
        colmean = groupby_clust[col].mean().rename(col + '_mean')
        colmean /= num_days

        df[col + '_nz'] = df[col].fillna(0).astype(bool)
        colnonzero = df.groupby(clust, as_index=False)[col + '_nz'].sum()
        colstats = cudf.merge(colmean, colnonzero, on=clust, how='outer')

        colmedian = df.groupby(clust, as_index=False)[col].median().rename(col + '_median')
        colmedian /= num_days
        colstats = cudf.merge(colstats, colmedian, on=clust, how='outer')
        clusters = cudf.merge(clusters, colstats, on=clust)

        # Compute mean only using non-zero values
        clusters[col + '_mean'] = clusters[col + '_mean'] * clusters['clust_size'] / clusters[col + '_nz']

        clusters[col + '_nz_total'] = df[col + '_nz'].sum()
        clusters[col + '_mean_total'] = df[col].sum() / (clusters[col + '_nz_total'] * num_days)
        clusters[col + '_median_total'] = df[col].median()
        clusters[col + '_mean_dev'] = clusters[col + '_mean'] / clusters[col + '_mean_total'] - 1
        clusters[col + '_median_dev'] = clusters[col + '_median'] / clusters[col + '_median_total'] - 1
        clusters[col + '_nz_frac'] = 100 * clusters[col + '_nz'] / clusters['clust_size']
        clusters[col + '_nz_frac_total'] = 100 * clusters[col + '_nz_total'] / df.shape[0]
        clusters[col + '_nz_frac_rem'] = 100 * (clusters[col + '_nz_total'] -
                                                clusters[col + '_nz']) / (df.shape[0] - clusters['clust_size'])

    devcols = [col for col in clusters.columns if col.endswith('_mean_dev')]
    clusters[devcols] = clusters[devcols].fillna(0)
    for idx in range(clusters.shape[0]):
        clust_num = clusters[clust].iloc[idx]
        if cluster_id != 'all':
            if clust_num != cluster_id:
                continue
        print("\nCLUSTER:{}, Size:{}".format(clust_num, clusters['clust_size'].iloc[idx]))
        coldev = clusters[devcols].iloc[idx]
        sortcols = coldev.abs().sort_values(ascending=False)

        cols = sortcols.iloc[:30].index.to_arrow().to_pylist()
        cols_ = [x[:-9] + '_nz_frac' for x in cols]
        subsetdf = clusters.iloc[idx][cols_]
        # Keep only features that have >5% non-zero values
        cols = subsetdf.loc[subsetdf > 5].index.to_arrow().to_pylist()
        cols = cols[:min(len(cols), top_diff_detail_feats)]
        cols_ = [x[:-8] for x in cols]
        cols_ = [x + y for x in cols_ for y in ('_mean', '_mean_total', '_nz_frac', '_nz_frac_rem')]

        print("Features with Top differences:\n{}\n".format(sortcols.iloc[:top_diff_summary_feats]))
        if verbose:
            print(clusters.iloc[idx][cols_])

        for col in cols:
            col = col[:-8]
            freq_0 = df.loc[df[clust] == 0][col].value_counts()
            freq_rem = df.loc[df[clust] != 0][col].value_counts()
            freq_0, freq_rem = 100 * freq_0 / freq_0.sum(), 100 * freq_rem / freq_rem.sum()
            freqs = cudf.merge(freq_0, freq_rem, left_index=True, right_index=True, how='outer')
            freqs.fillna(0, inplace=True)
            density_diff = freqs[col + '_x'] - freqs[col + '_y']
            density_diff = density_diff.abs()
            if density_diff.max() > 5:
                print("{}: Max diff={:.2f}%".format(col, density_diff.max()))

    if write_differences:
        imp_cols = [
            'cluster_dbscan_eps0.0005_minkp1',
            'domain_ctr_validate_src_cnt_mean',
            'domain_ctr_validate_src_cnt_nz_frac',
            'total_logins_src_cnt_mean',
            'total_logins_src_cnt_nz_frac',
            'logon_type_2_mean',
            'logon_type_2_nz_frac',
            'logon_type_5_mean',
            'logon_type_5_nz_frac',
            'logon_type_11_mean',
            'logon_type_11_nz_frac',
            'logon_type_10_mean',
            'logon_type_10_nz_frac',
            'total_user_initi_logoff_cnt_nz_frac',
            'DomainName_cnt_nz_frac',
            'UserName_cnt_mean',
            'DomainName_cnt_mean',
            'TGT_req_src_cnt_mean',
            'TGS_req_src_cnt_mean',
            'uname_that_compacnt_login_frac_nz_frac',
            'uname_that_compacnt_login_frac_mean',
            'total_logoff_cnt_mean',
            'total_logoff_cnt_nz_frac'
        ]
        clusters[imp_cols].to_csv('../results/clust_chars.csv', header=True, index=False)
    return clusters


def fit_tsne(df, init='random'):
    """Fit a Sklearn TSNE model on the DataFrame df

    Returns the TSNE transformed input.
    """
    tsne = skmani.TSNE(n_components=2, learning_rate=100, init=init)
    return tsne.fit_transform(df)


def fit_tsne_cuml(df, perplexity=25.0, learning_rate=100.0):
    """Fit a cuml TSNE model on the DataFrame df

    Returns the TSNE transformed input.
    """
    tsne = cuml.TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate)
    return tsne.fit_transform(df)


def tsneplot_util(df, tsne_cols, color_map, title, clust):
    """
    Utility to scatter in 2-dimension for datapoints in df DataFrame
    where TSNE transformed values are given in tsne_cols. If tsne_cols
    size is more than 2, remaining values are ignored in the scatter plot.
    """
    tsne1, tsne2 = tsne_cols[0], tsne_cols[1]
    df['color'] = 'k'
    for k, v in color_map.items():
        df.loc[df[clust] == k, 'color'] = v
    scatter = plt.scatter(tsne1, tsne2, c='color', data=df)
    plt.xlabel('tSNE1')
    plt.ylabel('tSNE2')
    plt.title(title)
