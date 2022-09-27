# Copyright (c) 2019-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  Apache-2.0
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


# Install required dependencies
#!pip install torch
#!pip install tldextract
#!pip install tensorflow

# Import required dependencies
import pandas as pd
import numpy as np
import tldextract
import json, os
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.core import Activation
from keras.layers.embeddings import Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Functions
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
def get_domain_space(domain):
    try:
        return " ".join(domain)
    except:
        return ""
def argmin_label(row):
    emb = np.array(row[[0,1,2,3,4,5,6,7,8,9]])
    list_dist = [np.sum(np.power(emb-ref_families[key],2)) for key in ref_families]
    arg_m = np.argmin(list_dist)
    dist_m = np.min(list_dist)
    if dist_m < 0.5:
        row['predict_family_label'] = list(families)[arg_m]
    else:
        row['predict_family_label'] = 'others'
    row['predict_family_dist'] = dist_m
    return row
def predict_family_df(domain):
    output_df = pd.DataFrame(columns = ['char_'+str(i) for i in range(75)] , data = domain)
    y_pred = model.predict(domain)
    output_df["predict"] = y_pred
    domain_emb = model_family.predict(domain)
    output_df = pd.concat([output_df, pd.DataFrame(domain_emb)], axis=1)
    output_df = output_df.apply(lambda row : argmin_label(row),axis=1)
    output_df.drop([0,1,2,3,4,5,6,7,8,9], axis=1, inplace=True)
    return output_df

# Loading binary model
model = tf.keras.models.load_model('../models/dga_model_keras')

# Loading family model
model_family = tf.keras.models.load_model('../models/dga_family_model_keras')

# Loading reference embedding vectors
ref_save_df = pd.read_csv('../models/model_ref_new.csv')

ref_families = {}
for dga_family in ref_save_df['Family']:
    ref_families[dga_family] = np.array(ref_save_df[ref_save_df['Family']==dga_family][['0','1','2','3','4','5','6','7','8','9']])
families = list(ref_families.keys())

# Read tokenizer
tokenizer = Tokenizer()
tokenizer.word_index = pd.read_csv('tokenizer.csv').set_index('keys')['values'].to_dict()

# Read the URL plugins
path = "../datasets/dga-appshield-data/"
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

data['Domain'] = data['URL'].apply(get_domain)
data_chars = data['Domain'].apply(get_domain_space)
data_tokens = tokenizer.texts_to_sequences(data_chars)
data_tokens = pad_sequences(data_tokens, maxlen=75, padding='post')
data_pred = predict_family_df(data_tokens)

# Create onnx models
#!pip install onnxruntime
#!pip install git+https://github.com/onnx/tensorflow-onnx
#!python -m tf2onnx.convert --saved-model /raid0/haim/haim/dga_model_keras --output dga_binary_model_tensorflow.onnx
#!python -m tf2onnx.convert --saved-model /raid0/haim/haim/dga_family_model_keras --output dga_family_model_tensorflow.onnx

# Inference onnx model
import onnx
import onnxruntime

ONNX_DGA_FILE_PATH = "ga_binary_model_tensorflow.onnx"
session = onnxruntime.InferenceSession(ONNX_DGA_FILE_PATH, None)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
result = session.run([output_name], {input_name: data_tokens.astype(np.float32)})

ONNX_DGA_FAMILY_FILE_PATH = "dga_family_model_tensorflow.onnx"
session = onnxruntime.InferenceSession(ONNX_DGA_FAMILY_FILE_PATH, None)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
result = session.run([output_name], {input_name: data_tokens.astype(np.float32)})



