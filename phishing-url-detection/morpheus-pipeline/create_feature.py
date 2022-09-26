# Copyright (c) 2022, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
import re
import typing
from urllib.parse import urlparse

import dataclasses
import typing

import numpy as np
import pandas as pd
import srf
import tldextract
from dask.distributed import Client
from morpheus.config import Config
from morpheus.messages import MultiMessage
from morpheus.pipeline.multi_message_stage import MultiMessageStage
from morpheus.pipeline.stream_pair import StreamPair
from morpheus.stages.input.appshield_source_stage import AppShieldMessageMeta
from srf.core import operators as ops

MAX_LEN = 500
STRUCTURAL_FEATURES = [
    'domain_in_alexa',
    'domain_len',
    'domain_numbers',
    'domain_isalnum',
    'subdomain_len',
    'subdomain_numbers_count',
    'subdomain_parts_count',
    'tld_len',
    'tld_parts_count',
    'queries_amount',
    'fragments_amount',
    'path_len',
    'path_slash_counts',
    'path_double_slash_counts',
    'brand_in_subdomain',
    'brand_in_path',
    'path_max_len'
]


def remove_prefix(text):
    try:
        if text.startswith('ftp://'):
            text = text[len('https://'):]
        if text.startswith('https://'):
            text = text[len('https://'):]
        if text.startswith('http://'):
            text = text[len('http://'):]
        if text.startswith('www.'):
            text = text[len('www.'):]
    except:
        text = ''
    return text


def clean(text):
    # strip '
    text = text.strip("'")
    # convert to lower letters
    text = text.lower()
    # remove punctuation marks
    text = re.sub('[^0-9a-zA-Z]+', ' ', text)
    # remove extra spaces
    text = re.sub(' +', ' ', text)
    # strip spaces
    text = text.strip(" ")
    return text


# Clean url with remove short and long words
def clean_nlp(text):
    text = clean(text)
    text = ' '.join([x for x in text.split(' ') if x.isnumeric() == False and len(x) > 1 and len(x) < 21])
    return text


def strip_se(url):
    return url.strip("'").strip('\n')


def add_http(url):
    if url.startswith('http'):
        return url
    return 'http://' + url


def get_domain(url):
    domain = tldextract.extract(url).domain
    if domain:
        return domain
    return ''


def get_domain(url):
    domain = tldextract.extract(url).domain
    if domain in ['ddns', 'bazar', 'onion']:
        url = url.strip('https://').strip('http://')
        urls = url.split('.')
        urls_i = urls.index(domain)
        if urls_i == 0:
            return domain
        return urls[urls_i - 1]
    return domain


def get_subdomain(url):
    subdomain = tldextract.extract(url).subdomain
    domain = tldextract.extract(url).domain
    if domain in ['ddns', 'bazar', 'onion']:
        url = url.strip('https://').strip('http://')
        urls = url.split('.')
        urls_i = urls.index(domain)
        if urls_i == 0:
            return subdomain
        return ".".join(urls[:urls_i - 1])
    return subdomain


def get_tld(url):
    tld = tldextract.extract(url).suffix
    domain = tldextract.extract(url).domain
    if domain in ['ddns', 'bazar', 'onion']:
        url = url.strip('https://').strip('http://')
        urls = url.split('.')
        urls_i = urls.index(domain)
        if urls_i == 0:
            return tld
        return ".".join(urls[urls_i:])
    return tld


def get_url_parsed(url):
    url_parsed = urlparse(url)
    return url_parsed


def get_path(url):
    url_parsed = urlparse(url)
    return url_parsed.path


def get_len(s):
    return len(s)


def get_count_numbers(s):
    return sum(c.isdigit() for c in s)


def get_not_alphanumeric(s):
    if s.isalnum() == True:
        return 1
    return 0


def get_count_parts(s):
    return len(s.split('.'))


def get_count_queries(s):
    url_parsed_query = urlparse(s).query
    if url_parsed_query == '':
        return 0
    return len(url_parsed_query.split('&'))


def get_count_fragments(s):
    url_parsed_fragment = urlparse(s).fragment
    if url_parsed_fragment == '':
        return 0
    return 1


def get_count_slash(s):
    return s.count('/')


def get_double_slash(s):
    return s.count('//')


def get_count_upper(s):
    return sum(1 for c in s if c.isupper())


def get_brand_in_subdomain(s):
    for brand in [
            'citibank',
            'whatsapp',
            'netflix',
            'dropbox',
            'wetransfer',
            'rakuten',
            'itau',
            'outlook',
            'ebay',
            'facebook',
            'hsbc',
            'linkedin',
            'instagram',
            'google',
            'paypal',
            'dhl',
            'alibaba',
            'bankofamerica',
            'apple',
            'microsoft',
            'skype',
            'amazon',
            'yahoo',
            'wellsfargo',
            'americanexpress'
    ]:
        if brand in s:
            return 1
    return 0


def get_brand_in_path(s):
    for brand in [
            'citibank',
            'whatsapp',
            'netflix',
            'dropbox',
            'wetransfer',
            'rakuten',
            'itau',
            'outlook',
            'ebay',
            'facebook',
            'hsbc',
            'linkedin',
            'instagram',
            'google',
            'paypal',
            'dhl',
            'alibaba',
            'bankofamerica',
            'apple',
            'microsoft',
            'skype',
            'amazon',
            'yahoo',
            'wellsfargo',
            'americanexpress'
    ]:
        if brand in s:
            return 1
    return 0


def get_domain_alexa(s, alexa_rank_1k_domain_unique, alexa_rank_100k_domain_unique):
    if s in alexa_rank_1k_domain_unique:
        return 2
    elif s in alexa_rank_100k_domain_unique:
        return 1
    return 0


def get_max_len_path(path_clean):
    if path_clean == '':
        return 0
    path_split = [len(f) for f in path_clean.split()]
    return np.max(path_split, 0)


# Calculating the features
def create_features(df, alexa_rank_1k_domain_unique, alexa_rank_100k_domain_unique):
    df['domain_in_alexa'] = df['Domain'].apply(
        lambda x: get_domain_alexa(x, alexa_rank_1k_domain_unique, alexa_rank_100k_domain_unique))
    df['domain_len'] = df['Domain'].apply(get_len)
    df['domain_numbers'] = df['Domain'].apply(get_count_numbers)
    df['domain_isalnum'] = df['Domain'].apply(get_not_alphanumeric)
    df['subdomain_len'] = df['Subdomain'].apply(get_len)
    df['subdomain_numbers_count'] = df['Subdomain'].apply(get_count_numbers)
    df['subdomain_parts_count'] = df['Subdomain'].apply(get_count_parts)
    df['tld_len'] = df['Tld'].apply(get_len)
    df['tld_parts_count'] = df['Tld'].apply(get_count_parts)
    df['url_len'] = df['URL'].apply(get_len)
    df['queries_amount'] = df['URL'].apply(get_count_queries)
    df['fragments_amount'] = df['URL'].apply(get_count_fragments)
    df['path_len'] = df['Path'].apply(get_len)
    df['path_slash_counts'] = df['Path'].apply(get_count_slash)
    df['path_double_slash_counts'] = df['Path'].apply(get_double_slash)
    df['brand_in_subdomain'] = df['Subdomain'].apply(get_brand_in_subdomain)
    df['brand_in_path'] = df['Path'].apply(get_brand_in_path)
    df['Path_clean'] = df['Path'].apply(lambda x: clean(x))
    df['path_max_len'] = df['Path_clean'].apply(get_max_len_path)
    return df


# Processing the url - domain, subdomain, tld, path and get URL's features
def processing(df, alexa_rank_1k_domain_unique, alexa_rank_100k_domain_unique):
    # strip url
    df['URL'] = df['URL'].apply(strip_se)
    # add http
    df['URL'] = df['URL'].apply(add_http)
    # get domain
    df['Domain'] = df['URL'].apply(get_domain)
    # get sub domain
    df['Subdomain'] = df['URL'].apply(get_subdomain)
    # get tld
    df['Tld'] = df['URL'].apply(get_tld)
    # get path
    df['Path'] = df['URL'].apply(get_path)
    # Create features
    df = create_features(df, alexa_rank_1k_domain_unique, alexa_rank_100k_domain_unique)
    return df


@dataclasses.dataclass
class FeatureConfig:
    required_plugins: typing.List[str]
    full_memory_address: int
    file_extn_list: typing.List[str]
    protections: typing.Dict[str, str]
    features_dummy_data: typing.Dict[str, int]


def _build_features(full_df: pd.DataFrame,
                    word_index,
                    df_max_min,
                    alexa_rank_1k_domain_unique,
                    alexa_rank_100k_domain_unique) -> pd.DataFrame:

    snapshot_df = full_df
    snapshot_df = processing(snapshot_df, alexa_rank_1k_domain_unique, alexa_rank_100k_domain_unique)

    snapshot_df['URL_clean'] = snapshot_df['URL'].copy().apply(remove_prefix)
    snapshot_df['URL_clean'] = snapshot_df['URL_clean'].apply(lambda x: clean_nlp(x))

    url_stractural_features = pd.DataFrame()

    for feature in STRUCTURAL_FEATURES:
        max_feature = df_max_min[feature].iloc[1]
        min_feature = df_max_min[feature].iloc[0]
        url_stractural_feature = snapshot_df[feature].copy()
        url_stractural_features[feature] = (url_stractural_feature - min_feature) / (max_feature - min_feature)
    
    words_df = snapshot_df.URL_clean.str.split(" ", expand=True)

    for col in words_df.columns:
        words_df[col] = words_df[col].map(word_index)

    words_df = words_df.fillna(0)
    
    pad_width = MAX_LEN - words_df.shape[1]

    padded_np_array = np.pad(
        words_df.to_numpy(), ((0, 0), (0, pad_width)), mode="constant"
    )

    df_features = pd.concat([url_stractural_features, pd.DataFrame(
        columns=["word_" + str(i) for i in range(MAX_LEN)], data=padded_np_array)], axis=1)
    
    df_features["URL_clean"] = snapshot_df["URL_clean"]
    df_features["URL"] = snapshot_df["URL"]

    return df_features


def _combined_features(x: typing.List[pd.DataFrame]) -> pd.DataFrame:

    return pd.concat(x)


class CreateFeatureURLStage(MultiMessageStage):

    def __init__(self, c: Config, required_plugins: typing.List[str], feature_columns: typing.List[str], tokenizer_path, max_min_norm_path, alexa_path):
        self._required_plugins = required_plugins
        self._feature_columns = feature_columns
        self._features_dummy_data = dict.fromkeys(self._feature_columns, 0)
        alexa_rank = pd.read_csv(alexa_path, header=None)
        alexa_rank.columns = ['index', 'url']
        alexa_rank_domain = alexa_rank['url'].apply(get_domain)
        self.alexa_rank_1k = alexa_rank_domain.iloc[0:1000]
        self.alexa_rank_100k = alexa_rank_domain.iloc[1000:100000]
        self.alexa_rank_1k_domain_unique = pd.unique(self.alexa_rank_1k)
        self.alexa_rank_100k_domain_unique = pd.unique(self.alexa_rank_100k)
        self.df_max_min = pd.read_csv(max_min_norm_path)
        self._word_index = pd.read_csv(tokenizer_path).set_index('keys')['values'].to_dict()
        self._feature_config = FeatureConfig(required_plugins=self._required_plugins,
                                             full_memory_address=0,
                                             file_extn_list=[],
                                             protections=[],
                                             features_dummy_data=self._features_dummy_data)

        self._client = Client(threads_per_worker=2, n_workers=6)

        super().__init__(c)

    @property
    def name(self) -> str:
        return "create-features"

    def accepted_types(self) -> typing.Tuple:
        """
        Returns accepted input types for this stage.

        """
        return (AppShieldMessageMeta, )
    
    def supports_cpp_node(self):
        return False

    def _build_single(self, seg: srf.Builder, input_stream: StreamPair) -> StreamPair:
        stream = input_stream[0]

        def node_fn(input: srf.Observable, output: srf.Subscriber):

            def on_next(x: AppShieldMessageMeta):
                to_send = []
                snapshot_fea_dfs = []

                df = x.df
                df['PID'] = df['PID'].astype(str)
                df['PID_Process'] = df.PID + '_' + df.Process
                snapshot_ids = x.df.snapshot_id.unique()

                all_dfs = [df[df.snapshot_id == snapshot_id] for snapshot_id in snapshot_ids]

                snapshot_fea_dfs = self._client.map(_build_features,
                                                    all_dfs,
                                                    word_index=self._word_index,
                                                    df_max_min=self.df_max_min,
                                                    alexa_rank_1k_domain_unique=self.alexa_rank_1k_domain_unique,
                                                    alexa_rank_100k_domain_unique=self.alexa_rank_100k_domain_unique,)

                features_df = self._client.submit(_combined_features, snapshot_fea_dfs)

                features_df = features_df.result()
                self._client.shutdown()
                features_df['source'] = x.source
                features_df['pid_process'] = df['PID_Process']
                features_df['snapshot_id'] = df['snapshot_id']
                features_df = features_df.sort_values(by=["pid_process", "snapshot_id"]).reset_index(drop=True)
                
                x = AppShieldMessageMeta(features_df, x.source)

                unique_pid_processes = features_df.pid_process.unique()

                # Create multi messaage per pid_process
                for pid_process in unique_pid_processes:
                    start = features_df.pid_process[features_df.pid_process == pid_process].index.min()
                    stop = features_df.pid_process[features_df.pid_process == pid_process].index.max() + 1
                    multi_message = MultiMessage(meta=x, mess_offset=start, mess_count=stop - start)
                    to_send.append(multi_message)
                return to_send

            def on_completed():
                # Close dask client when pipeline initiates shutdown
                self._client.close()

            input.pipe(ops.map(on_next), ops.on_completed(on_completed), ops.flatten()).subscribe(output)

        node = seg.make_node_full(self.unique_name, node_fn)
        seg.make_edge(stream, node)
        stream = node
        return stream, MultiMessage
