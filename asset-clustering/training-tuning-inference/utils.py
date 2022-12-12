import os
import logging
import numpy as np
import pandas as pd
import cudf
import cupy as cp

def read_netflow(fname, nrows=None):
    df = cudf.read_csv(fname, nrows=nrows)
    netflow_header = [
        'time', 'Duration', 'SrcDevice','DstDevice', 'Protocol', 'SrcPort',
        'DstPort', 'SrcPackets', 'DstPackets', 'SrcBytes', 'DstBytes']
    df.columns = netflow_header
    df['time_h'] = cudf.to_datetime(df['time'],unit='ms')
    return df


def read_wls(fname, file_path=False, nrows=None):
    """Read the windows event log file and return a data frame with the data
    """
    if file_path:
        df = cudf.read_json(fname, lines=True, nrows=nrows)
    else:
        txt = "\n".join([x.decode("utf-8") for x in fname])
        df = cudf.read_json(txt, lines=True)

    df['time_dt'] = cudf.to_datetime(df['Time'], unit='s')  # format='%Y-%m-%d %H:%M:%S.%f')
    return df


def compute_username_cnt(df_, host_, srcdict_):
    df_ = df_[['LogHost', 'UserName']].copy()
    df_ = df_.loc[~df_['UserName'].isna()]

    unique_usernames = df_.groupby('LogHost')['UserName'].agg('unique')
    unique_usernames = unique_usernames.rename('unique_usernames').to_pandas()

    for i in range(unique_usernames.shape[0]):
        hostval, unames = unique_usernames.index[i], unique_usernames.iloc[i]
        srcdict_['Unames'][hostval] = srcdict_['Unames'][hostval].union(unames)

    uname_cnt_df= cudf.DataFrame({
        'LogHost':srcdict_['Unames'].keys(),
        'UserName_cnt':[len(v) for v in srcdict_['Unames'].values()]})

    comb = cudf.merge(host_['UserName_cnt'].reset_index(),
                      uname_cnt_df, how='outer', on='LogHost')

    # DomainName_cnt_x has DomaiName counts upto prev chunk, DomainName_cnt_y has
    # updated DomaiName counts only for hosts present in new data.
    comb.loc[~comb['UserName_cnt_y'].isna(), 'UserName_cnt_x'] = 0
    comb = comb.fillna({'UserName_cnt_y':0})

    comb['UserName_cnt'] = comb['UserName_cnt_x'] + comb['UserName_cnt_y']
    comb = comb.drop(['UserName_cnt_x', 'UserName_cnt_y'], axis=1).set_index('LogHost')
    host_ = host_.drop('UserName_cnt', axis=1)
    host_ = cudf.merge(host_, comb, how='inner', on='LogHost')

    return host_, srcdict_


def compute_username_domain_cnt(df_, host_, srcdict_):
    df_ = df_[['LogHost', 'DomainName']].copy()
    df_ = df_.loc[~df_['DomainName'].isna()]

    unique_username_domains = df_.groupby('LogHost')['DomainName'].agg('unique')
    unique_username_domains = unique_username_domains.rename('unique_username_domains').to_pandas()

    for i in range(unique_username_domains.shape[0]):
        hostval, unames = unique_username_domains.index[i], unique_username_domains.iloc[i]
        srcdict_['UserDomains'][hostval] = srcdict_['UserDomains'][hostval].union(unames)

    udomain_cnt_df= cudf.DataFrame({
        'LogHost': srcdict_['UserDomains'].keys(),
        'DomainName_cnt':[len(v) for v in srcdict_['UserDomains'].values()]})
    udomain_cnt_df = udomain_cnt_df.set_index('LogHost', drop=True)

    comb = cudf.merge(host_['DomainName_cnt'].reset_index(),
                      udomain_cnt_df, how='outer', on='LogHost')

    # DomainName_cnt_x has DomaiName counts upto prev chunk, DomainName_cnt_y has
    # updated DomaiName counts only for hosts present in new data.
    comb.loc[~comb['DomainName_cnt_y'].isna(), 'DomainName_cnt_x'] = 0
    comb = comb.fillna({'DomainName_cnt_y': 0})

    comb['DomainName_cnt'] = comb['DomainName_cnt_x'] + comb['DomainName_cnt_y']
    comb = comb.drop(['DomainName_cnt_x', 'DomainName_cnt_y'], axis=1).set_index('LogHost')

    host_ = host_.drop('DomainName_cnt', axis=1)
    host_ = cudf.merge(host_, comb, how='inner', on='LogHost')

    host_ = cudf.merge(host_, udomain_cnt_df, how='outer', on='LogHost')
    host_ = host_.drop(['DomainName_cnt_x'], axis=1)
    host_ = host_.rename({'DomainName_cnt_y': 'DomainName_cnt'}, axis=1)
    return host_, srcdict_


def account_logons(df_, host_):
    df_4634 = df_.loc[df_['EventID'] == 4634]
    num_logons = df_4634['LogHost'].value_counts().rename_axis('LogHost').rename('num_accnt_logons')

    df_4624 = df_.loc[df_['EventID'] == 4624]
    num_succ_logons = df_4624['LogHost'].value_counts().rename_axis('LogHost').rename('num_accnt_succ_logons')

    num_logons = pd.merge(num_logons, num_succ_logons, on='LogHost', how='outer')
    num_logons = num_logons.fillna(0)
    if set(num_logons.index)-set(host_.index):
        logging.error("Found extra LogHosts. UNEXPECTED BEHAVIOR")

    host_ = pd.merge(host_, num_logons, how='outer', on='LogHost', )
    host_['num_accnt_succ_logons'] = host_['num_accnt_succ_logons_x'] + host_['num_accnt_succ_logons_y']
    host_['num_accnt_logons'] = host_['num_accnt_logons_x'] + host_['num_accnt_logons_y']
    host_.drop(['num_accnt_logons_x', 'num_accnt_logons_y',
                'num_accnt_succ_logons_x', 'num_accnt_succ_logons_y'], axis=1, inplace=True)
    return host_


def logon_types(df_, host_, valid_logon_types):
    """
    Computes number of logins by each LogonType
    """
    def cnt_logontypes(df_touse, logon_types, host_, suffix=''):
        for ltype in logon_types:
            col_name = 'logon_type_{}{}'.format(suffix, int(ltype))
            df_ltype = df_touse.loc[df_touse['LogonType'] == ltype]
            dfltype_cnt = df_ltype['LogHost'].value_counts().rename(col_name)
            dfltype_cnt.index.rename('LogHost', inplace=True)
            host_ = cudf.merge(host_, dfltype_cnt, on='LogHost', how='left')
            host_[col_name] = host_[col_name + '_x'] + host_[col_name + '_y']
            host_.drop([col_name + '_x', col_name + '_y'], axis=1, inplace=True)
        return host_

    df_ = df_.loc[df_['EventID'].isin([4624, 4625])]
    ltype_indf = df_['LogonType'].unique()
    logon_types_ = ltype_indf.loc[ltype_indf.isin(valid_logon_types)].to_pandas().to_list()
    host_ = cnt_logontypes(df_, logon_types_, host_)

    df_src = df_.loc[~df_['Source'].isna()]
    ltype_indf = df_src['LogonType'].unique()
    logon_types_ = ltype_indf.loc[ltype_indf.isin(valid_logon_types)].to_pandas().to_list()
    host_ = cnt_logontypes(df_src, logon_types_, host_, suffix='frm_')

    return host_


def compute_diff_source_logon_cnt(df_, host_, srcdict_):
    """
    For each LogHost, Computes total number of unique sources with some event
    Looks at all EventTypes.
    """

    df_ = df_[['LogHost', 'Source']].copy()
    df_ = df_.loc[~df_['Source'].isna()]

    unique_sources = df_.groupby('LogHost')['Source'].agg('unique')
    unique_sources = unique_sources.rename('unique_sources').to_pandas()

    for i in range(unique_sources.shape[0]):
        hostval, srces = unique_sources.index[i], unique_sources.iloc[i]
        srcdict_['Sources'][hostval] = srcdict_['Sources'][hostval].union(srces)

    src_cnt_df= cudf.DataFrame({
        'LogHost': srcdict_['Sources'].keys(),
        'Source_cnt': [len(v) for v in srcdict_['Sources'].values()]})

    comb = cudf.merge(host_['Source_cnt'].reset_index(), src_cnt_df, how='outer', on='LogHost')

    # Source_cnt_x has source counts from prev, Source_cnt_y has updated source
    #  counts only for hosts present in new data.
    comb.loc[~comb['Source_cnt_y'].isna(), 'Source_cnt_x'] = 0
    comb = comb.fillna({'Source_cnt_y':0})

    comb['Source_cnt'] = comb['Source_cnt_x'] + comb['Source_cnt_y']
    comb = comb.drop(['Source_cnt_x', 'Source_cnt_y'], axis=1).set_index('LogHost')
    host_ = host_.drop('Source_cnt', axis=1)
    host_ = cudf.merge(host_, comb, how='inner', on='LogHost')

    return host_, srcdict_


def compute_logins_with_loghostuname(df_, host_):
    """
    Computes logins from the username corresponding to
    a. computer accounts corresp. to specified LogHost i.e. UserName= LogHost+'$'
    b. computer accounts corresp. to other LogHost i.e. UserName ending with $ and != LogHost+'$'
    """
    df_ = df_.loc[df_['EventID'].isin([4624, 4625])]
    df_1 = df_.loc[(df_['UserName'].str.endswith('$')) & (df_['UserName'] != df_['LogHost']+'$')]

    uname_other_compacnt_login_cnt = df_1['LogHost'].value_counts()\
                                                 .rename('uname_other_compacnt_login_cnt')

    uname_other_compacnt_login_cnt.index.rename('LogHost', inplace=True)
    host_ = cudf.merge(host_, uname_other_compacnt_login_cnt, how='outer', on='LogHost')

    df_2 = df_.loc[df_['UserName'] == df_['LogHost']+'$']
    uname_that_compacnt_login_cnt = df_2['LogHost'].value_counts()\
                                                 .rename('uname_that_compacnt_login_cnt')
    uname_that_compacnt_login_cnt.index.rename('LogHost', inplace=True)
    host_ = cudf.merge(host_, uname_that_compacnt_login_cnt, how='outer', on='LogHost')

    for col in ['uname_other_compacnt_login_cnt', 'uname_that_compacnt_login_cnt']:
        host_[col] = host_[col+'_x'] + host_[col+'_y']
        host_.drop([col + '_x', col + '_y'], axis=1, inplace=True)
    return host_


def compute_eventid_cnt(df_, evid_, ev_str_, host_):
    df_evid = df_.loc[df_['EventID'] == evid_]
    event_cnt = df_evid['LogHost'].value_counts().rename(ev_str_)
    event_cnt.index.rename('LogHost', inplace=True)

    if set(event_cnt.index.to_pandas())-set(host_.index.to_pandas()):
        pdb.set_trace()
        logging.error("Found extra LogHosts. UNEXPECTED BEHAVIOR")
    host_ = cudf.merge(host_, event_cnt, how='left', on='LogHost')
    host_[ev_str_] = host_[ev_str_ + '_x'] + host_[ev_str_ + '_y']
    host_.drop([ev_str_ + '_y', ev_str_ + '_x'], axis=1, inplace=True)

    return host_


def compute_eventid_cnt_source(df_, evid_, ev_str_, host_):
    """For each asset=i, Counts the number of rows with
     EventID == evid_ &
     Source == i
    and assigns host_[i][ev_str_] += count
    """
    df_evid = df_.loc[df_['EventID'] == evid_]
    event_cnt = df_evid['Source'].value_counts().rename(ev_str_)
    event_cnt.index.rename('LogHost', inplace=True)

    if set(event_cnt.index.to_pandas())-set(host_.index.to_pandas()):
        pdb.set_trace()
        logging.error("Found extra LogHosts. UNEXPECTED BEHAVIOR")
    host_ = cudf.merge(host_, event_cnt, how='left', on='LogHost')
    host_[ev_str_] = host_[ev_str_ + '_x'] + host_[ev_str_ + '_y']
    host_.drop([ev_str_ + '_y', ev_str_ + '_x'], axis=1, inplace=True)

    return host_


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


def hist_util(df0, col, clust_, num_bins=8):
    coldf = df0.dropna(axis=0, subset=[col])
    col_clust0, col_clustrest = coldf.loc[coldf[clust_] == 0, col], coldf.loc[coldf[clust_] != 0, col]

    val_25pct, val_75pct = np.percentile(col_clust0.loc[col_clust0!=0], [25, 75])

    binw = (val_75pct - val_25pct)*2/num_bins
    bins = [val_25pct+i*binw for i in range(-num_bins//4,3*num_bins//4 +1)]
    clust0_hist, clust0_vals = np.histogram(col_clust0, bins=num_bins)
    clustrem_hist, clustrem_vals =  np.histogram(col_clustrest, bins=clust0_vals)

    clust0_hist, clustrem_hist = cp.asnumpy(clust0_hist), cp.asnumpy(clustrem_hist)

    return clust0_hist, clustrem_hist, bins


def compute_val_counts(df_, col, clust_):
    freq_0 = df_.loc[df_[clust_] == 0][col].value_counts()
    freq_rem = df_.loc[df_[clust_] != 0][col].value_counts()

    freq_0, freq_rem = 100*freq_0/freq_0.sum(), 100*freq_rem/freq_rem.sum()
    freqs =  pd.merge(freq_0, freq_rem, left_index=True,
                      right_index=True, how='outer')
    freqs.fillna(0, inplace=True)
    return freqs


def compute_chars(df_, clust_, num_days=1, cluster_id='all',
                  write_differences=False, verbose=False,
                  top_diff_summary_feats=10,
                  top_diff_detail_feats=8):
    pddf = df_.to_pandas()
    clusters = df_[clust_].value_counts().rename('clust_size')
    clusters = clusters.reset_index().rename({'index':clust_}, axis=1)
    ignore_cols = [clust_,'LogHost', 'num_accnt_logons','num_accnt_succ_logons']
    for col in set(df_.columns)-set(ignore_cols):
        colmean = df_.groupby(clust_, as_index=False)[col].mean().rename(col+'_mean')
        colmean /= num_days

        df_[col + '_nz'] =df_[col].fillna(0).astype(bool)
        colnonzero = df_.groupby(clust_, as_index=False)[col+'_nz'].sum()
        colstats = cudf.merge(colmean, colnonzero, on=clust_, how='outer')

        colmedian = df_.groupby(clust_, as_index=False)[col].median().rename(col+'_median')
        colmedian /= num_days
        colstats = cudf.merge(colstats, colmedian, on=clust_, how='outer')
        clusters = cudf.merge(clusters, colstats, on=clust_)

        #Compute mean only using non-zero values
        clusters[col + '_mean'] = clusters[col + '_mean'] * clusters['clust_size']/ clusters[col + '_nz']

        clusters[col+'_nz_total'] = df_[col+'_nz'].sum()
        clusters[col+'_mean_total'] = df_[col].sum()/(clusters[col+'_nz_total']*num_days)
        clusters[col+'_median_total'] = df_[col].median()
        clusters[col+'_mean_dev'] = clusters[col+'_mean']/clusters[col+'_mean_total'] - 1
        clusters[col+'_median_dev'] = clusters[col+'_median']/clusters[col+'_median_total'] - 1
        clusters[col+'_nz_frac'] = 100*clusters[col+'_nz']/clusters['clust_size']
        clusters[col+'_nz_frac_total'] = 100*clusters[col+'_nz_total']/df_.shape[0]
        clusters[col+'_nz_frac_rem'] = 100*(clusters[col+'_nz_total']-clusters[col+'_nz'])/(df_.shape[0]-clusters['clust_size'])

    devcols = [col for col in clusters.columns if col.endswith('_mean_dev')]
    clusters[devcols] = clusters[devcols].fillna(0)
    for idx in range(clusters.shape[0]):
        clust_num = clusters[clust_].iloc[idx]
        if cluster_id != 'all':
            if clust_num != cluster_id:
                continue
        print("\nCLUSTER:{}, Size:{}".format(clust_num, clusters['clust_size'].iloc[idx]))
        coldev = clusters[devcols].iloc[idx]
        sortcols = coldev.abs().sort_values(ascending=False)

        cols = sortcols.iloc[:30].index.to_arrow().to_pylist()
        cols_ = [x[:-9]+'_nz_frac' for x in cols]
        subsetdf = clusters.iloc[idx][cols_]
        # Keep only features that have >5% non-zero values
        cols = subsetdf.loc[subsetdf > 5].index.to_arrow().to_pylist()
        cols = cols[:min(len(cols), top_diff_detail_feats)]
        cols_ = [x[:-8] for x in cols]
        cols_ = [x+y for x in cols_ for y in ('_mean', '_mean_total', '_nz_frac', '_nz_frac_rem')]

        print("Features with Top differences:\n{}\n".format(
            sortcols.iloc[:top_diff_summary_feats]))
        if verbose:
            print(clusters.iloc[idx][cols_])
        if not verbose:
            return

        for col in cols:
            col = col[:-8]
            freq_0 = pddf.loc[pddf[clust_]==0][col].value_counts()
            freq_rem = pddf.loc[pddf[clust_]!=0][col].value_counts()
            freq_0, freq_rem = 100*freq_0/freq_0.sum(), 100*freq_rem/freq_rem.sum()
            freqs =  cudf.merge(freq_0, freq_rem, left_index=True, right_index=True, how='outer')
            freqs.fillna(0, inplace=True)
            density_diff = freqs[col+'_x'] - freqs[col+'_y']
            density_diff = density_diff.abs()
            if density_diff.max() > 5:
                print("{}: Max diff={:.2f}%".format(col, density_diff.max()))

    if write_differences:
        imp_cols = [
            'cluster_dbscan_eps0.0005_minkp1','domain_ctr_validate_src_cnt_mean',
            'domain_ctr_validate_src_cnt_nz_frac',
            'total_logins_src_cnt_mean', 'total_logins_src_cnt_nz_frac',
            'logon_type_2_mean','logon_type_2_nz_frac',
            'logon_type_5_mean','logon_type_5_nz_frac',
            'logon_type_11_mean', 'logon_type_11_nz_frac',
            'logon_type_10_mean', 'logon_type_10_nz_frac',
            'total_user_initi_logoff_cnt_nz_frac', 'DomainName_cnt_nz_frac',
            'UserName_cnt_mean', 'DomainName_cnt_mean',
            'TGT_req_src_cnt_mean', 'TGS_req_src_cnt_mean',
            'uname_that_compacnt_login_frac_nz_frac', 'uname_that_compacnt_login_frac_mean',
            'total_logoff_cnt_mean', 'total_logoff_cnt_nz_frac']
        clusters[imp_cols].to_csv('../results/clust_chars.csv', header=True, index=False)
    return clusters

