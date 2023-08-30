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

import numpy as np
import scipy.sparse as sp
import scipy.io as scio
import pandas as pd
import glob
from collections import defaultdict
import networkx as nx
import os
from itertools import islice


def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))


def load_sflow(path="../dataset"):
    '''Ingests the sflow data, enrich with Armis data and returns features, adjacency matrix, graph object'''

    print('Loading from raw data file...')
    arista_csvs = glob.glob(os.path.join(f'{path}/arista_sflow/', "*.csv"))
    arista_df = pd.concat((pd.read_csv(f) for f in arista_csvs), ignore_index=True)
    cumulus_csvs = glob.glob(os.path.join(f'{path}/cumulus_sflow/', "*.csv"))
    cumulus_df = pd.concat((pd.read_csv(f) for f in cumulus_csvs), ignore_index=True)

    # Read Armis device data
    armis_data_device = pd.read_csv(f'{path}/armis_enrichment_device_v3.csv')
    armis_data_device['mac_address'] = armis_data_device['mac_address'].str.lower()
    armis_data_device['mac_address'] = armis_data_device['mac_address'].str.replace(" ", "")

    # Read Armis app data
    armis_data_app = pd.read_csv(f'{path}/armis_enrichment_app_v3.csv')
    armis_data_app['mac_address'] = armis_data_app['mac_address'].str.lower()

    # Armis data has multiple devices linked to same mac, hence separating and getting them independently
    for i in range(0, 5):
        armis_data_device['mac_address_' + str(i)] = None
    temp_res = armis_data_device['mac_address'].apply(lambda x: x.split(','))
    for index, row in armis_data_device.iterrows():
        for i in range(min(5, len(temp_res[index]))):
            armis_data_device.loc[index, 'mac_address_' + str(i)] = temp_res[index][i]

    armis_data_device['category'].fillna(value='Unknown', inplace=True)
    armis_data_device['operatingSystem'].fillna(value='Unknown', inplace=True)
    armis_data_device['name'].fillna(value='Unknown', inplace=True)
    armis_data_device['manufacturer'].fillna(value='Unknown', inplace=True)
    armis_data_device['mtype'].fillna(value='Unknown', inplace=True)

    # Acces the port dictionary to create port features from sflow data
    port_df = pd.read_csv(f'{path}/ports.txt', sep='\t')
    port_df.columns = port_df.columns.str.replace(' ', '')
    cols = port_df.select_dtypes(object).columns
    port_df[cols] = port_df[cols].apply(lambda x: x.str.replace(' ', ''))
    # create port dict
    port_dict = {}
    for rowno, row in port_df.iterrows():
        port_dict[str(row['PortNumber'])] = str(row['Description']) + '_Port_' + str(row['PortNumber'])

    # Combine the data and assign
    cumulus_df['sflow_type'] = 'cumulus'
    arista_df['sflow_type'] = 'arista'

    # Concat sflow data from cumulus and arista
    sflow_raw = pd.concat([cumulus_df, arista_df], ignore_index=True)
    print(sflow_raw.shape)
    #print(sflow_raw.head(5))

    # Unique mac addresses
    mac_set = set(sflow_raw.SRC_MAC)
    mac_set.update(sflow_raw.DST_MAC)
    unique_macs = list(mac_set)
    len(unique_macs)

    # Create mac dict
    mac_dict = defaultdict(lambda: len(mac_dict))
    mac_list = [mac_dict[n] for n in unique_macs]

    print('unique source mac : {}'.format(sflow_raw.SRC_MAC.nunique()))
    print('unique dest mac : {}'.format(sflow_raw.DST_MAC.nunique()))

    print("Calculating adjacency matrix")
    adj = defaultdict(lambda: len(adj))
    for raw_mac, mac_id in mac_dict.items():
        # Identify mac devices where source = destination
        loop_mac_df = sflow_raw.loc[(sflow_raw.SRC_MAC == raw_mac) | (sflow_raw.DST_MAC == raw_mac)]
        loop_mac_list = list(set(list(loop_mac_df.SRC_MAC.unique()) + list(loop_mac_df.DST_MAC.unique())))

        # Remove self relation from adj
        loop_mac_list = [x for x in loop_mac_list if x != raw_mac]
        adj[mac_id] = loop_mac_list

    n_items = take(5, adj.items())
    print(n_items)

    for k, v in adj.items():
        mac_id_list = []
        for raw_mac in v:
            mac_id_list.append(mac_dict[raw_mac])
        adj[k] = mac_id_list

    n_items = take(5, adj.items())
    print(n_items)

    # Create graph object
    G = nx.from_dict_of_lists(adj)
    adj = nx.adjacency_matrix(G)
    print("adj type : {}".format(type(adj)))

    # Construct feature matrix
    feat_dict = defaultdict(lambda: len(unique_macs))
    rows = []
    for raw_mac, mac_id in mac_dict.items():
        # print("\n macid : {}, raw_mac : {}".format(mac_id, raw_mac))
        mac_df = sflow_raw.loc[sflow_raw.SRC_MAC == raw_mac]

        count_of_host_sflows = mac_df.shape[0] if mac_df.shape[0] > 0 else np.nan

        sys_ports_df = mac_df[mac_df['SRC_PORT'].between(0, 1023, inclusive='both')] if \
        mac_df[mac_df['SRC_PORT'].between(0, 1023, inclusive='both')].shape[0] > 0 else pd.DataFrame()
        count_host_no_of_sys_ports = sys_ports_df['SRC_PORT'].nunique() if sys_ports_df.shape[0] > 0 else np.nan
        # std_host_sys_port = sys_ports_df['SRC_PORT'].std() if sys_ports_df.shape[0] > 0 else np.nan
        host_freq_sys_port = sys_ports_df['SRC_PORT'].mode()[0] if sys_ports_df.shape[0] > 0 else np.nan
        # count_host_freq_sys_port = sys_ports_df['SRC_PORT'].value_counts().values.tolist()[0] if sys_ports_df.shape[0] > 0 else np.nan

        user_ports_df = mac_df[mac_df['SRC_PORT'].between(1024, 49151, inclusive='both')] if \
        mac_df[mac_df['SRC_PORT'].between(1024, 49151, inclusive='both')].shape[0] > 0 else pd.DataFrame()
        count_host_no_of_user_ports = user_ports_df['SRC_PORT'].nunique() if user_ports_df.shape[0] > 0 else np.nan
        # std_host_user_port = user_ports_df['SRC_PORT'].std() if user_ports_df.shape[0] > 0 else np.nan
        host_freq_user_port = user_ports_df['SRC_PORT'].mode()[0] if user_ports_df.shape[0] > 0 else np.nan
        # count_host_freq_user_port = user_ports_df['SRC_PORT'].value_counts().values.tolist()[0] if user_ports_df.shape[0] > 0 else np.nan

        dynamic_ports_df = mac_df[mac_df['SRC_PORT'].between(49152, 65535, inclusive='both')] if \
        mac_df[mac_df['SRC_PORT'].between(49152, 65535, inclusive='both')].shape[0] > 0 else pd.DataFrame()
        count_host_no_of_dynamic_ports = dynamic_ports_df['SRC_PORT'].nunique(
        ) if dynamic_ports_df.shape[0] > 0 else np.nan

        other_sys_ports_df = mac_df[mac_df['DST_PORT'].between(0, 1023, inclusive='both')] if \
        mac_df[mac_df['DST_PORT'].between(0, 1023, inclusive='both')].shape[0] > 0 else pd.DataFrame()
        count_other_no_of_sys_ports = other_sys_ports_df['DST_PORT'].nunique(
        ) if other_sys_ports_df.shape[0] > 0 else np.nan
        # std_other_sys_port = other_sys_ports_df['DST_PORT'].std() if other_sys_ports_df.shape[0] > 0 else np.nan
        other_freq_sys_port = other_sys_ports_df['DST_PORT'].mode()[0] if other_sys_ports_df.shape[0] > 0 else np.nan
        # count_other_freq_sys_port = other_sys_ports_df['DST_PORT'].value_counts().values.tolist()[0] if other_sys_ports_df.shape[0] > 0 else np.nan

        other_user_ports_df = mac_df[mac_df['DST_PORT'].between(1024, 49151, inclusive='both')] if \
        mac_df[mac_df['DST_PORT'].between(1024, 49151, inclusive='both')].shape[0] > 0 else pd.DataFrame()
        count_other_no_of_user_ports = other_user_ports_df['DST_PORT'].nunique(
        ) if other_user_ports_df.shape[0] > 0 else np.nan
        # std_other_user_port = other_user_ports_df['DST_PORT'].std() if other_user_ports_df.shape[0] > 0 else np.nan
        other_freq_user_port = other_user_ports_df['DST_PORT'].mode()[0] if other_user_ports_df.shape[0] > 0 else np.nan
        # count_other_freq_user_port = other_user_ports_df['DST_PORT'].value_counts().values.tolist()[0] if other_user_ports_df.shape[0] > 0 else np.nan

        other_dynamic_ports_df = mac_df[mac_df['DST_PORT'].between(49152, 65535, inclusive='both')] if \
        mac_df[mac_df['DST_PORT'].between(49152, 65535, inclusive='both')].shape[0] > 0 else pd.DataFrame()
        count_other_no_of_dynamic_ports = other_dynamic_ports_df['DST_PORT'].nunique(
        ) if other_dynamic_ports_df.shape[0] > 0 else np.nan

        count_host_port_smaller_other = mac_df.loc[mac_df.SRC_PORT < mac_df.DST_PORT].shape[0] if \
        mac_df.loc[mac_df.SRC_PORT < mac_df.DST_PORT].shape[0] > 0 else np.nan

        avg_bytes_per_flow = mac_df.BYTES.mean() if mac_df.shape[0] > 0 else np.nan
        max_bytes_over_flows = mac_df.BYTES.max() if mac_df.shape[0] > 0 else np.nan

        avg_packets_per_flow = mac_df.PACKETS.mean() if mac_df.shape[0] > 0 else np.nan
        max_packets_over_flows = mac_df.PACKETS.max() if mac_df.shape[0] > 0 else np.nan

        count_tcp_flow = mac_df[mac_df.PROTOCOL == 'tcp'].shape[0] if mac_df[mac_df.PROTOCOL ==
                                                                             'tcp'].shape[0] > 0 else np.nan
        count_udp_flow = mac_df[mac_df.PROTOCOL == 'udp'].shape[0] if mac_df[mac_df.PROTOCOL ==
                                                                             'udp'].shape[0] > 0 else np.nan
        # count_igmp_flow = mac_df[mac_df.PROTOCOL=='igmp'].shape[0] if mac_df[mac_df.PROTOCOL=='igmp'].shape[0] > 0 else np.nan
        count_icmp_flow = mac_df[mac_df.PROTOCOL == 'icmp'].shape[0] if mac_df[mac_df.PROTOCOL ==
                                                                               'icmp'].shape[0] > 0 else np.nan
        count_otherProto_flow = mac_df[~(mac_df.PROTOCOL.isin(['tcp', 'udp', 'igmp', 'icmp']))].shape[0] if \
        mac_df[~(mac_df.PROTOCOL.isin(['tcp', 'udp', 'igmp', 'icmp']))].shape[0] > 0 else np.nan
        '''
        mac_port_dict = {}
        for rowno, row in port_df.iterrows():
            mac_port_df = mac_df[(mac_df.SRC_PORT==row['PortNumber']) | (mac_df.DST_PORT==row['PortNumber'])]
            mac_port_dict[port_dict[str(row['PortNumber'])]] = mac_port_df.shape[0] if mac_port_df.shape[0] > 0 else np.nan
        '''

        # Port features
        # oracle_db_port = mac_df[(mac_df.SRC_PORT.isin([2483, 2484])) | (mac_df.DST_PORT.isin([2483, 2484]))].shape[0] if mac_df[(mac_df.SRC_PORT.isin([2483, 2484])) | (mac_df.DST_PORT.isin([2483, 2484]))].shape[0] > 0 else np.nan
        ftp_ports = mac_df[(mac_df.SRC_PORT.isin([20, 21])) | (mac_df.DST_PORT.isin([20, 21]))].shape[0] if \
        mac_df[(mac_df.SRC_PORT.isin([20, 21])) | (mac_df.DST_PORT.isin([20, 21]))].shape[0] > 0 else np.nan
        ssh_ports = mac_df[(mac_df.SRC_PORT.isin([22])) | (mac_df.DST_PORT.isin([22]))].shape[0] if \
        mac_df[(mac_df.SRC_PORT.isin([22])) | (mac_df.DST_PORT.isin([22]))].shape[0] > 0 else np.nan
        dns_ports = mac_df[(mac_df.SRC_PORT.isin([53])) | (mac_df.DST_PORT.isin([53]))].shape[0] if \
        mac_df[(mac_df.SRC_PORT.isin([53])) | (mac_df.DST_PORT.isin([53]))].shape[0] > 0 else np.nan
        http_ports = mac_df[(mac_df.SRC_PORT.isin([80])) | (mac_df.DST_PORT.isin([80]))].shape[0] if \
        mac_df[(mac_df.SRC_PORT.isin([80])) | (mac_df.DST_PORT.isin([80]))].shape[0] > 0 else np.nan
        ntp_ports = mac_df[(mac_df.SRC_PORT.isin([123])) | (mac_df.DST_PORT.isin([123]))].shape[0] if \
        mac_df[(mac_df.SRC_PORT.isin([123])) | (mac_df.DST_PORT.isin([123]))].shape[0] > 0 else np.nan
        bgp_ports = mac_df[(mac_df.SRC_PORT.isin([179])) | (mac_df.DST_PORT.isin([179]))].shape[0] if \
        mac_df[(mac_df.SRC_PORT.isin([179])) | (mac_df.DST_PORT.isin([179]))].shape[0] > 0 else np.nan
        net_bios_ports = mac_df[(mac_df.SRC_PORT.isin([137, 139])) | (mac_df.DST_PORT.isin([137, 139]))].shape[0] if \
        mac_df[(mac_df.SRC_PORT.isin([137, 139])) | (mac_df.DST_PORT.isin([137, 139]))].shape[0] > 0 else np.nan
        https_ports = mac_df[(mac_df.SRC_PORT.isin([443])) | (mac_df.DST_PORT.isin([443]))].shape[0] if \
        mac_df[(mac_df.SRC_PORT.isin([443])) | (mac_df.DST_PORT.isin([443]))].shape[0] > 0 else np.nan
        isakmp_ports = mac_df[(mac_df.SRC_PORT.isin([500])) | (mac_df.DST_PORT.isin([500]))].shape[0] if \
        mac_df[(mac_df.SRC_PORT.isin([500])) | (mac_df.DST_PORT.isin([500]))].shape[0] > 0 else np.nan
        ftps_ports = mac_df[(mac_df.SRC_PORT.isin([989, 990])) | (mac_df.DST_PORT.isin([989, 990]))].shape[0] if \
        mac_df[(mac_df.SRC_PORT.isin([989, 990])) | (mac_df.DST_PORT.isin([989, 990]))].shape[0] > 0 else np.nan
        sql_ports = mac_df[(mac_df.SRC_PORT.isin([1433, 1434, 2483, 2484, 3306, 5432, 5434])) | (
            mac_df.DST_PORT.isin([1433, 1434, 2483, 2484, 3306, 5432, 5434]))].shape[0] if mac_df[
                (mac_df.SRC_PORT.isin([1433, 1434, 2483, 2484, 3306, 5432, 5434])) |
                (mac_df.DST_PORT.isin([1433, 1434, 2483, 2484, 3306, 5432, 5434]))].shape[0] > 0 else np.nan
        # oracle_db_port = mac_df[(mac_df.SRC_PORT.isin([2483, 2484])) | (mac_df.DST_PORT.isin([2483, 2484]))].shape[0] if mac_df[(mac_df.SRC_PORT.isin([2483, 2484])) | (mac_df.DST_PORT.isin([2483, 2484]))].shape[0] > 0 else np.nan
        rdp_ports = mac_df[(mac_df.SRC_PORT.isin([3389])) | (mac_df.DST_PORT.isin([3389]))].shape[0] if \
        mac_df[(mac_df.SRC_PORT.isin([3389])) | (mac_df.DST_PORT.isin([3389]))].shape[0] > 0 else np.nan
        dtgramUdp_ports = mac_df[(mac_df.SRC_PORT.isin([1900])) | (mac_df.DST_PORT.isin([1900]))].shape[0] if \
        mac_df[(mac_df.SRC_PORT.isin([1900])) | (mac_df.DST_PORT.isin([1900]))].shape[0] > 0 else np.nan
        multiclassdns_ports = mac_df[(mac_df.SRC_PORT.isin([5353])) | (mac_df.DST_PORT.isin([5353]))].shape[0] if \
        mac_df[(mac_df.SRC_PORT.isin([5353])) | (mac_df.DST_PORT.isin([5353]))].shape[0] > 0 else np.nan
        smbUdp_ports = mac_df[(mac_df.SRC_PORT.isin([445])) | (mac_df.DST_PORT.isin([445]))].shape[0] if \
        mac_df[(mac_df.SRC_PORT.isin([445])) | (mac_df.DST_PORT.isin([445]))].shape[0] > 0 else np.nan

        # armis feats
        device_armis_data = armis_data_device.loc[(armis_data_device.mac_address_0 == raw_mac) |
                                                  (armis_data_device.mac_address_1 == raw_mac) |
                                                  (armis_data_device.mac_address_2 == raw_mac) |
                                                  (armis_data_device.mac_address_3 == raw_mac) |
                                                  (armis_data_device.mac_address_4 == raw_mac)]
        device_category = device_armis_data.category.iloc[0] if device_armis_data.shape[0] > 0 else ''
        computers_device_category_armis = 1 if device_category == 'Computers' else np.nan
        networkequip_device_category_armis = 1 if device_category == 'Network Equipment' else np.nan
        automations_device_category_armis = 1 if device_category == 'Automations' else np.nan

        device_mtype = device_armis_data.mtype.iloc[0] if device_armis_data.shape[0] > 0 else ''
        servers_device_mtype_armis = 1 if device_mtype == 'Servers' else np.nan
        pc_device_mtype_armis = 1 if device_mtype == 'Personal Computers' else np.nan
        hypervisor_device_mtype_armis = 1 if device_mtype == 'Hypervisor' else np.nan
        vm_device_mtype_armis = 1 if device_mtype == 'Virtual Machines' else np.nan
        switches_device_mtype_armis = 1 if device_mtype == 'Switches' else np.nan
        routers_device_mtype_armis = 1 if device_mtype == 'Routers' else np.nan
        # pdus_device_mtype_armis = 1 if device_mtype=='PDUs' else np.nan
        # singleBoardcomp_device_mtype_armis = 1 if device_mtype=='Single-Board Computers' else np.nan

        device_os = device_armis_data.operatingSystem.iloc[0] if device_armis_data.shape[0] > 0 else ''
        ubuntu_device_os_armis = 1 if device_os == 'Ubuntu' else np.nan
        windows_device_os_armis = 1 if 'Windows' in device_os else np.nan
        linux_device_os_armis = 1 if 'Linux' in device_os or 'Debian' in device_os or 'CentOS' in device_os else np.nan
        vmwareESXi_device_os_armis = 1 if 'VMware' in device_os else np.nan

        device_manufacturer = device_armis_data.manufacturer.iloc[0] if device_armis_data.shape[0] > 0 else ''
        supermicro_device_manufacturer_armis = 1 if 'Supermicro' in device_manufacturer else np.nan
        huawei_device_manufacturer_armis = 1 if 'Huawei' in device_manufacturer else np.nan
        shenzenLianrui_device_manufacturer_armis = 1 if 'Shenzhen Lianrui' in device_manufacturer else np.nan
        wistron_device_manufacturer_armis = 1 if 'Wistron' in device_manufacturer else np.nan
        gigabyte_device_manufacturer_armis = 1 if 'Giga-Byte Technology' in device_manufacturer else np.nan
        fujitsu_device_manufacturer_armis = 1 if 'Fujitsu' in device_manufacturer else np.nan
        vmware_device_manufacturer_armis = 1 if 'VMware' in device_manufacturer else np.nan
        arista_device_manufacturer_armis = 1 if 'Arista Networks' in device_manufacturer else np.nan
        mellanox_device_manufacturer_armis = 1 if 'Mellanox Technologies' in device_manufacturer else np.nan
        dell_device_manufacturer_armis = 1 if 'Dell' in device_manufacturer else np.nan
        intel_device_manufacturer_armis = 1 if 'Intel' in device_manufacturer else np.nan

        app_armis = armis_data_app.loc[armis_data_app.mac_address == raw_mac]
        #print(app_armis.info())
        app_device = app_armis.stack().groupby(level=0).apply(' '.join).iloc[0] if app_armis.shape[0] > 0 else ''
        openBSD_app_device_armis = 1 if 'OpenBSD' in app_device else np.nan
        bind_app_device_armis = 1 if 'bind' in app_device else np.nan
        apache_app_device_armis = 1 if 'Apache' in app_device else np.nan
        fireEyeSecurity_app_device_armis = 1 if 'FireEye' in app_device else np.nan
        sql_app_device_armis = 1 if 'sql' in app_device or 'SQL' in app_device else np.nan
        adobe_app_device_armis = 1 if 'Adobe' in app_device else np.nan
        kuberntes_app_device_armis = 1 if 'kubernetes' in app_device else np.nan
        microsoft_app_device_armis = 1 if 'Microsoft' in app_device or 'Windows' in app_device or 'Azure' in app_device or 'microsoft' in app_device else np.nan
        sambaAdmin_app_device_armis = 1 if 'samba' in app_device else np.nan
        publickeyx509_app_device_armis = 1 if 'x.509_certificate' in app_device else np.nan
        ngix_app_device_armis = 1 if 'NGINX' in app_device or 'nginx' in app_device else np.nan
        browser_app_device_armis = 1 if 'Chrome' in app_device or 'Edge' in app_device or 'Browser' in app_device else np.nan
        db_app_device_armis = 1 if 'mongodb' in app_device or 'cassandra' in app_device or 'Redis' in app_device else np.nan
        python_app_device_armis = 1 if 'python' in app_device else np.nan
        vmware_app_device_armis = 1 if 'VMware' in app_device or 'Carbon Black' in app_device else np.nan

        feat_dict = {
            'raw_mac': raw_mac,
            'mac_id': mac_id,
            'count_of_host_sflows': count_of_host_sflows,
            'count_host_no_of_sys_ports': count_host_no_of_sys_ports,  # 'std_host_sys_port' : std_host_sys_port,
            'host_freq_sys_port': host_freq_sys_port,  # 'count_host_freq_sys_port' : count_host_freq_sys_port,
            'count_host_no_of_user_ports': count_host_no_of_user_ports,  # 'std_host_user_port': std_host_user_port,
            'host_freq_user_port': host_freq_user_port,  # 'count_host_freq_user_port': count_host_freq_user_port,
            'count_host_no_of_dynamic_ports': count_host_no_of_dynamic_ports,
            # 'count_other_no_of_sys_ports': count_other_no_of_sys_ports,
            # 'std_other_sys_port': std_other_sys_port,
            # 'other_freq_sys_port': other_freq_sys_port,
            # 'count_other_freq_sys_port' : count_other_freq_sys_port,
            # 'count_other_no_of_user_ports': count_other_no_of_user_ports,
            # 'std_other_user_port': std_other_user_port,
            # 'other_freq_user_port': other_freq_user_port,
            # 'count_other_freq_user_port': count_other_freq_user_port,
            # 'count_other_no_of_dynamic_ports': count_other_no_of_dynamic_ports,
            # 'count_host_port_smaller_other': count_host_port_smaller_other,
            'avg_bytes_per_flow': avg_bytes_per_flow,
            'max_bytes_over_flows': max_bytes_over_flows,
            'avg_packets_per_flow': avg_packets_per_flow,
            'max_packets_over_flows': max_packets_over_flows,
            'count_tcp_flow': count_tcp_flow,
            'count_udp_flow': count_udp_flow,  # 'count_igmp_flow': count_igmp_flow,
            'count_icmp_flow': count_icmp_flow,
            'count_otherProto_flow': count_otherProto_flow,
            'ftp_ports': ftp_ports,
            'ssh_ports': ssh_ports,
            'dns_ports': dns_ports,
            'http_ports': http_ports,
            'ntp_ports': ntp_ports,
            'bgp_ports': bgp_ports,
            'net_bios_ports': net_bios_ports,
            'https_ports': https_ports,
            'isakmp_ports': isakmp_ports,
            'ftps_ports': ftps_ports,
            'sql_ports': sql_ports,
            'rdp_ports': rdp_ports,
            'dtgramUdp_ports': dtgramUdp_ports,
            'multiclassdns_ports': multiclassdns_ports,
            'smbUdp_ports': smbUdp_ports,
            'computers_device_category_armis': computers_device_category_armis,
            'networkequip_device_category_armis': networkequip_device_category_armis,
            # 'automations_device_category_armis' : automations_device_category_armis,
            'servers_device_mtype_armis': servers_device_mtype_armis,
            'pc_device_mtype_armis': pc_device_mtype_armis,
            'hypervisor_device_mtype_armis': hypervisor_device_mtype_armis,
            'vm_device_mtype_armis': vm_device_mtype_armis,
            'switches_device_mtype_armis': switches_device_mtype_armis,
            'routers_device_mtype_armis': routers_device_mtype_armis,
            'ubuntu_device_os_armis': ubuntu_device_os_armis,
            'windows_device_os_armis': windows_device_os_armis,
            'linux_device_os_armis': linux_device_os_armis,
            'vmwareESXi_device_os_armis': vmwareESXi_device_os_armis,
            'supermicro_device_manufacturer_armis': supermicro_device_manufacturer_armis,
            'huawei_device_manufacturer_armis': huawei_device_manufacturer_armis,
            'shenzenLianrui_device_manufacturer_armis': shenzenLianrui_device_manufacturer_armis,
            'wistron_device_manufacturer_armis': wistron_device_manufacturer_armis,
            'gigabyte_device_manufacturer_armis': gigabyte_device_manufacturer_armis,
            'fujitsu_device_manufacturer_armis': fujitsu_device_manufacturer_armis,
            'vmware_device_manufacturer_armis': vmware_device_manufacturer_armis,
            'arista_device_manufacturer_armis': arista_device_manufacturer_armis,
            'mellanox_device_manufacturer_armis': mellanox_device_manufacturer_armis,
            'dell_device_manufacturer_armis': dell_device_manufacturer_armis,
            'intel_device_manufacturer_armis': intel_device_manufacturer_armis,
            'openBSD_app_device_armis': openBSD_app_device_armis,
            'bind_app_device_armis': bind_app_device_armis,
            'apache_app_device_armis': apache_app_device_armis,
            'fireEyeSecurity_app_device_armis': fireEyeSecurity_app_device_armis,
            'sql_app_device_armis': sql_app_device_armis,
            'adobe_app_device_armis': adobe_app_device_armis,
            'kuberntes_app_device_armis': kuberntes_app_device_armis,
            'microsoft_app_device_armis': microsoft_app_device_armis,
            'sambaAdmin_app_device_armis': sambaAdmin_app_device_armis,
            'publickeyx509_app_device_armis': publickeyx509_app_device_armis,
            'ngix_app_device_armis': ngix_app_device_armis,
            'browser_app_device_armis': browser_app_device_armis,
            'db_app_device_armis': db_app_device_armis,
            'python_app_device_armis': python_app_device_armis,
            'vmware_app_device_armis': vmware_app_device_armis
        }
        # feat_dict.update(mac_port_dict)
        rows.append(feat_dict)
    feat_df = pd.DataFrame(rows)
    # remove null columns and columns with 1 unique values
    feat_df = feat_df.loc[:, ~(feat_df.isna().all())]
    # feat_df = feat_df.loc[:, feat_df.nunique() > 1]

    sdf = feat_df.iloc[:, 2:].astype(pd.SparseDtype("float", np.nan))
    features = sdf.sparse.to_coo().tocsr()

    return features.toarray(), adj.toarray(), feat_df, mac_dict, sflow_raw, G, adj


def load_data(name, data_path):
    if name.lower() == 'sflow':
        features, adj, features_df, mac_dict, sflow_raw, G, adj_raw = load_sflow(data_path)
        return features, adj, features_df, mac_dict, sflow_raw, G, adj_raw
    path = 'data/{}.mat'.format(name)
    data = scio.loadmat(path)
    adj = data['W']
    return data['X'], adj