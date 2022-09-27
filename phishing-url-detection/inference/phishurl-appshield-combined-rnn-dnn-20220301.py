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

import json
import os
import re

from urllib.parse import urlparse

import numpy as np
import pandas as pd
import tensorflow as tf
import tldextract

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Constants
MAX_LEN= 500
THRESHOLD = 0.976 # Precision = 0.9997 and Recall = 0.605
STRACTURAL_FEATURES = ['domain_in_alexa','domain_len','domain_numbers','domain_isalnum','subdomain_len','subdomain_numbers_count',
            'subdomain_parts_count','tld_len','tld_parts_count','queries_amount','fragments_amount',
            'path_len','path_slash_counts','path_double_slash_counts','brand_in_subdomain','brand_in_path','path_max_len']

# Functions for reading URL jsons
def convert_json_to_df(f,file,plugin_type):
    data = json.load(f)
    features_plugin = data["titles"]
    if plugin_type=='vadinfo':
        features_plugin.remove('SHA256')
    plugin_df = pd.DataFrame(columns=features_plugin, data=data["data"])
    plugin_df['snapshot'] = int(file.split("/")[-2].split('-')[1])
    return plugin_df
def get_plugin_files(files):
    urls_df = pd.DataFrame()
    timestamp = ""
    for file in files:
        file_ext = file.split('/')[-1]
        with open(file,encoding='latin1') as f:
            try:
                urls_df_i = convert_json_to_df(f, file, "urls")
                urls_df = urls_df.append(urls_df_i)
                file_ext_dot = file_ext.split('.')
                file_ext_dot_dash = file_ext_dot[0].split("_")
                timestamp = file_ext_dot_dash[1]+"_"+file_ext_dot_dash[2]
            except:
                    print("error: "+file)
    return urls_df, timestamp

# Functions for cleanind the URL for NLP processing
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
    text = ' '.join([x for x in text.split(' ') if x.isnumeric()==False and len(x)>1 and len(x)<21])
    return text

# Functions for disassemble the URL
def strip_se(url):
    return url.strip("'").strip('\n')

def add_http(url):
    if url.startswith('http'):
        return url
    return 'http://'+url

def get_domain(url):
    domain = tldextract.extract(url).domain
    if domain:
        return domain
    return ''
def get_domain(url):
    domain = tldextract.extract(url).domain
    if domain in ['ddns','bazar','onion']:
        url = url.strip('https://').strip('http://')
        urls = url.split('.')
        urls_i = urls.index(domain)
        if urls_i == 0:
            return domain
        return urls[urls_i-1]
    return domain

def get_subdomain(url):
    subdomain = tldextract.extract(url).subdomain
    domain = tldextract.extract(url).domain
    if domain in ['ddns','bazar','onion']:
        url = url.strip('https://').strip('http://')
        urls = url.split('.')
        urls_i = urls.index(domain)
        if urls_i == 0:
            return subdomain
        return ".".join(urls[:urls_i-1])
    return subdomain

def get_tld(url):
    tld = tldextract.extract(url).suffix
    domain = tldextract.extract(url).domain
    if domain in ['ddns','bazar','onion']:
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

# Functions for model features
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
    print(url_parsed_query.split('&'))
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
    for brand in ['citibank','whatsapp','netflix','dropbox','wetransfer','rakuten','itau','outlook','ebay','facebook','hsbc','linkedin','instagram','google','paypal','dhl','alibaba','bankofamerica','apple','microsoft','skype','amazon','yahoo','wellsfargo','americanexpress']:
        if brand in s:
            return 1
    return 0
def get_brand_in_path(s):
    for brand in ['citibank','whatsapp','netflix','dropbox','wetransfer','rakuten','itau','outlook','ebay','facebook','hsbc','linkedin','instagram','google','paypal','dhl','alibaba','bankofamerica','apple','microsoft','skype','amazon','yahoo','wellsfargo','americanexpress']:
        if brand in s:
            return 1
    return 0
def get_domain_alexa(s):
    if s in alexa_rank_1k_domain_unique:
        return 2
    elif s in alexa_rank_100k_domain_unique:
        return 1
    return 0
def get_max_len_path(path_clean):
    if path_clean == '':
        return 0
    path_split = [len(f) for f in path_clean.split()]
    return np.max(path_split,0)

# Calculating the features
def create_features(df):
    df['domain_in_alexa'] = df['Domain'].swifter.apply(get_domain_alexa)
    df['domain_len'] = df['Domain'].swifter.apply(get_len)
    df['domain_numbers'] = df['Domain'].swifter.apply(get_count_numbers)
    df['domain_isalnum'] = df['Domain'].swifter.apply(get_not_alphanumeric)
    df['subdomain_len'] = df['Subdomain'].swifter.apply(get_len)
    df['subdomain_numbers_count'] = df['Subdomain'].swifter.apply(get_count_numbers)
    df['subdomain_parts_count'] = df['Subdomain'].swifter.apply(get_count_parts)
    df['tld_len'] = df['Tld'].swifter.apply(get_len)
    df['tld_parts_count'] = df['Tld'].swifter.apply(get_count_parts)
    df['url_len'] = df['URL'].swifter.apply(get_len)
    df['queries_amount'] = df['URL'].swifter.apply(get_count_queries)
    df['fragments_amount'] = df['URL'].swifter.apply(get_count_fragments)
    df['path_len'] = df['Path'].swifter.apply(get_len)
    df['path_slash_counts'] = df['Path'].swifter.apply(get_count_slash)
    df['path_double_slash_counts'] = df['Path'].swifter.apply(get_double_slash)
    df['upper_amount'] = df['URL'].swifter.apply(get_count_upper)
    df['brand_in_subdomain'] = df['Subdomain'].swifter.apply(get_brand_in_subdomain)
    df['brand_in_path'] = df['Path'].swifter.apply(get_brand_in_path)
    df['Path_clean'] = df['Path'].swifter.apply(lambda x: clean(x))
    df['path_max_len'] = df['Path_clean'].swifter.apply(get_max_len_path)
    return df
# Processing the url - domain, subdomain, tld, path and get URL's features
def processing(df):
    # strip url
    df['URL'] = df['URL'].apply(strip_se)
    # add http
    df['URL'] = df['URL'].apply(add_http)
    #df['url'].apply(get_url_parsed)
    # get domain
    df['Domain'] = df['URL'].apply(get_domain)
    # get sub domain
    df['Subdomain'] = df['URL'].apply(get_subdomain)
    # get tld
    df['Tld'] = df['URL'].apply(get_tld)
    # get path
    df['Path'] = df['URL'].apply(get_path)
    # Create features
    df = create_features(df)
    return df

# Alexa rank dict
alexa_rank = pd.read_csv('../datasets/alexa-top-500k.csv',header=None)
alexa_rank.columns = ['index','url']
alexa_rank_domain = alexa_rank['url'].apply(get_domain)
alexa_rank_1k = alexa_rank_domain.iloc[0:1000]
alexa_rank_100k = alexa_rank_domain.iloc[1000:100000]
alexa_rank_1k_domain_unique = pd.unique(alexa_rank_1k)
alexa_rank_100k_domain_unique = pd.unique(alexa_rank_100k)

# Read the URL plugins
path = "../datasets/url-data/"
snapshots = os.listdir(path)
snapshots = [int(x.split('-')[1]) for x in snapshots if 'snap' in x]
snapshots.sort()
snapshots = ['snapshot-'+str(x) for x in snapshots]
for snap_file in snapshots:
    print(path+snap_file+"/")
    snap_file = path+snap_file+"/"
    files = []
    for r, d, f in os.walk(snap_file):
        for file in f:
            if '.json' in file:
                files.append(os.path.join(r, file))
    #try:
        data, timestamp = get_plugin_files(files)

url_df = processing(data)
url_df['URL_clean'] = url_df['URL'].copy().apply(remove_prefix)
url_df['URL_clean'] = url_df['URL_clean'].apply(lambda x: clean_nlp(x))

df_max_min = pd.read_csv('max_min_urls.csv')

url_stractural_features = url_df[STRACTURAL_FEATURES]
for feature in STRACTURAL_FEATURES:
    max_feature = df_max_min[feature][1]
    min_feature = df_max_min[feature][0]
    url_stractural_features[feature] = (url_stractural_features[feature] - min_feature) / (max_feature - min_feature)

# Read tokenizer
tokenizer = Tokenizer()
tokenizer.word_index = pd.read_csv('tokenizer_urls.csv').set_index('keys')['values'].to_dict()

url_df_clean = url_df['URL_clean']
url_clean_tokens = tokenizer.texts_to_sequences(url_df_clean)
url_clean_tokens = pad_sequences(url_clean_tokens, maxlen=MAX_LEN, padding='post')

df_output = pd.concat([url_stractural_features, pd.DataFrame(columns = ['word_'+str(i) for i in range(MAX_LEN)] , data = url_clean_tokens)], axis=1)

# Loading model
model = tf.keras.models.load_model('../models/url_model_keras')

url_stractural_features = np.array(url_stractural_features)
Y_pred = model.predict([url_clean_tokens, url_stractural_features])
df_output['pred'] = Y_pred

# Create onnx model
#!pip install onnxruntime
#!pip install git+https://github.com/onnx/tensorflow-onnx
#!python -m tf2onnx.convert --saved-model /raid0/haim/haim/url_model_keras_final --output url_model_tensorflow.onnx

# Inference onnx model
import onnx
import onnxruntime

ONNX_URL_FILE_PATH = "../models/url_model_tensorflow.onnx"
session = onnxruntime.InferenceSession(ONNX_URL_FILE_PATH, None)
input_name_0 = session.get_inputs()[0].name
input_name_1 = session.get_inputs()[1].name
output_name = session.get_outputs()[0].name
result = session.run([output_name], {input_name_0: url_clean_tokens.astype(np.float32), input_name_1: url_stractural_features.astype(np.float32)})