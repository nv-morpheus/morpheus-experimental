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

import re
from urllib.parse import urlparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tldextract
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Stractural feature of the url
ADDITIONAL_FEATURES = ['domain_in_alexa','domain_len','domain_numbers','domain_isalnum','subdomain_len','subdomain_numbers_count',
            'subdomain_parts_count','tld_len','tld_parts_count','queries_amount','fragments_amount',
            'path_len','path_slash_counts','path_double_slash_counts','brand_in_subdomain','brand_in_path','path_max_len']
# Max words in each url
MAX_LEN= 500
# Number of words in nlp model
NLP_TOKENS = 2000
# Number of epochs
NUM_EPOCHS = 1 # 150
# Size eof batch
BATCH_SIZE = 2000
# Size of embedding layer
EMBEDDING_DIM = 16
# Classes weight
CLASS_WEIGHTS = {0: 4000, 1:1}

# Clean url text
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
# Strip ' ' and '\n'
def strip_se(url):
    return url.strip("'").strip('\n')
# Add 'http://' for url if needed
def add_http(url):
    if url.startswith('http'):
        return url
    return 'http://'+url
# Get domain
def get_domain(url):
    domain = tldextract.extract(url).domain
    if domain:
        return domain
    return ''
# Get subdomain
def get_subdomain(url):
    subdomain = tldextract.extract(url).subdomain
    if subdomain:
        return subdomain
    return ''
# Get tld
def get_tld(url):
    tld = tldextract.extract(url).suffix
    if tld:
        return tld
    return ''
# Parse the url
def get_url_parsed(url):
    url_parsed = urlparse(url)
    return url_parsed
# Get url's path
def get_path(url):
    url_parsed = urlparse(url)
    return url_parsed.path
# Get url len
def get_len(s):
    return len(s)
# Get count of nubers in input
def get_count_numbers(s):
    return sum(c.isdigit() for c in s)
# Check if input is alpha-numeric
def get_not_alphanumeric(s):
    if s.isalnum() == True:
        return 1
    return 0
# Get count of dots
def get_count_parts(s):
    return len(s.split('.'))
# Get count of queries
def get_count_queries(s):
    url_parsed_query = urlparse(s).query
    if url_parsed_query == '':
        return 0
    return len(url_parsed_query.split('&'))
# Get count of fragments
def get_count_fragments(s):
    url_parsed_fragment = urlparse(s).fragment
    if url_parsed_fragment == '':
        return 0
    return 1
# Get count of slash
def get_count_slash(s):
    return s.count('/')
# Get count of double slash
def get_double_slash(s):
    return s.count('//')
# Get count of upper letters
def get_count_upper(s):
    return sum(1 for c in s if c.isupper())
# Check if brand in subdomain
def get_brand_in_subdomain(s):
    for brand in ['whatsapp','netflix','dropbox','wetransfer','rakuten','itau','outlook','ebay','facebook','hsbc','linkedin','instagram','google','paypal','dhl','alibaba','bankofamerica','apple','microsoft','skype','amazon','yahoo','wellsfargo','americanexpress']:
        if brand in s:
            return 1
    return 0
# Check if brand in path
def get_brand_in_path(s):
    for brand in ['whatsapp','netflix','dropbox','wetransfer','rakuten','itau','outlook','ebay','facebook','hsbc','linkedin','instagram','google','paypal','dhl','alibaba','bankofamerica','apple','microsoft','skype','amazon','yahoo','wellsfargo','americanexpress']:
        if brand in s:
            return 1
    return 0
# Check if domain is in Alexa rank
def get_domain_alexa(s):
    if s in alexa_rank_1k_domain_unique:
        return 2
    elif s in alexa_rank_100k_domain_unique:
        return 1
    return 0
# Get max of parts length
def get_max_len_path(path_clean):
    if path_clean == '':
        return 0
    path_split = [len(f) for f in path_clean.split()]
    return np.max(path_split,0)
# Check path empty
def check_path_empty(path):
    if path.strip("/") == "":
        return 1
    return 0

# Calculating the features
def create_features(df):
    df['domain_in_alexa'] = df['domain'].swifter.apply(get_domain_alexa)
    df['domain_len'] = df['domain'].swifter.apply(get_len)
    df['domain_numbers'] = df['domain'].swifter.apply(get_count_numbers)
    df['domain_isalnum'] = df['domain'].swifter.apply(get_not_alphanumeric)
    df['subdomain_len'] = df['subdomain'].swifter.apply(get_len)
    df['subdomain_numbers_count'] = df['subdomain'].swifter.apply(get_count_numbers)
    df['subdomain_parts_count'] = df['subdomain'].swifter.apply(get_count_parts)
    df['tld_len'] = df['tld'].swifter.apply(get_len)
    df['tld_parts_count'] = df['tld'].swifter.apply(get_count_parts)
    df['url_len'] = df['url'].swifter.apply(get_len)
    df['queries_amount'] = df['url'].swifter.apply(get_count_queries)
    df['fragments_amount'] = df['url'].swifter.apply(get_count_fragments)
    df['path_len'] = df['path'].swifter.apply(get_len)
    df['path_slash_counts'] = df['path'].swifter.apply(get_count_slash)
    df['path_double_slash_counts'] = df['path'].swifter.apply(get_double_slash)
    df['upper_amount'] = df['url'].swifter.apply(get_count_upper)
    df['brand_in_subdomain'] = df['subdomain'].swifter.apply(get_brand_in_subdomain)
    df['brand_in_path'] = df['path'].swifter.apply(get_brand_in_path)
    url_df['path_clean'] = url_df['path'].swifter.apply(lambda x: clean(x))
    url_df['path_max_len'] = url_df['path_clean'].swifter.apply(get_max_len_path)
    url_df['path_empty'] = df['path'].swifter.apply(check_path_empty)
    return df

# Processing the url - domain, subdomain, tld, path and get URL's features
def processing(df):
    # strip url
    df['url'] = df['url'].apply(strip_se)
    # add http
    df['url'] = df['url'].apply(add_http)
    # df['url'].apply(get_url_parsed)
    # get domain
    df['domain'] = df['url'].apply(get_domain)
    # get sub domain
    df['subdomain'] = df['url'].apply(get_subdomain)
    # get tld
    df['tld'] = df['url'].apply(get_tld)
    # get path
    df['path'] = df['url'].apply(get_path)
    # Create features
    df = create_features(df)
    return df

def data_preprocessing(df):
    df = processing(df)
    df['url_clean'] = df['url_clean'].apply(lambda x: clean_nlp(x))
    df['url_clean'] = df['url_clean'].apply(lambda x: clean_nlp(x))
    X = df[['url', 'url_clean'] + ADDITIONAL_FEATURES + ['label']]
    # Split the data for malicious and benign
    X_mal = X[X['label'] == 1]
    X_ben = X[X['label'] == 0]
    Y_mal = X_mal.pop('label')
    Y_ben = X_ben.pop('label')
    # Split the data to train and test
    X_mal_train, X_mal_test, Y_mal_train, Y_mal_test = train_test_split(X_mal, Y_mal, train_size=0.25)
    X_ben_train, X_ben_test, Y_ben_train, Y_ben_test = train_test_split(X_ben, Y_ben, train_size=0.8)
    X_train = X_mal_train.append(X_ben_train)
    Y_train = Y_mal_train.append(Y_ben_train)
    X_test = X_mal_test.append(X_ben_test)
    Y_test = Y_mal_test.append(Y_ben_test)
    return X_train, Y_train, X_test, Y_test

def stractural_processing(X_train, X_test):
    # Train and test features dataframe
    X_train_features = X_train[ADDITIONAL_FEATURES]
    X_test_features = X_test[ADDITIONAL_FEATURES]

    max_dict = {}
    min_dict = {}

    # Normalize the features
    for feature in X_train_features.columns:
        max_dict[feature] = X_train_features[feature].max()
        min_dict[feature] = X_train_features[feature].min()
        X_test_features[feature] = (X_test_features[feature] - X_train_features[feature].min()) / (
                    X_train_features[feature].max() - X_train_features[feature].min())
        X_train_features[feature] = (X_train_features[feature] - X_train_features[feature].min()) / (
                    X_train_features[feature].max() - X_train_features[feature].min())

    df_max_min = pd.DataFrame(columns=max_dict.keys())
    df_max_min = df_max_min.append(min_dict, ignore_index=True)
    df_max_min = df_max_min.append(max_dict, ignore_index=True)
    return X_train_features, X_test_features, df_max_min

def nlp_processing(X_train, X_test):
    # Train and test nlp dataframe
    X_train_nlp = X_train['url_clean']
    X_test_nlp = X_test['url_clean']
    # Convert the words to tokens
    tokenizer = Tokenizer(num_words=NLP_TOKENS)

    tokenizer.fit_on_texts(X_train_nlp)
    vocab_length = tokenizer.num_words + 1

    X_train_nlp = tokenizer.texts_to_sequences(X_train_nlp)
    X_test_nlp = tokenizer.texts_to_sequences(X_test_nlp)

    X_train_nlp = pad_sequences(X_train_nlp, maxlen=MAX_LEN, padding='post')
    X_test_nlp = pad_sequences(X_test_nlp, maxlen=MAX_LEN, padding='post')
    tokenizer_df = pd.DataFrame()
    tokenizer_df['keys'] = list(tokenizer.word_index.keys())[0:NLP_TOKENS]
    tokenizer_df['values'] = list(tokenizer.word_index.values())[0:NLP_TOKENS]
    return X_train_nlp, X_test_nlp, tokenizer_df, vocab_length

def train_model(X_train_nlp, X_train_features, Y_train):
    # Defining the model
    inputA = tf.keras.layers.Input(shape=(X_train_nlp.shape[1],))
    inputB = tf.keras.layers.Input(shape=(X_train_features.shape[1],))
    # First input will process the url text
    x = tf.keras.layers.Embedding(vocab_length, EMBEDDING_DIM, input_length=MAX_LEN)(inputA)
    x = tf.keras.layers.LSTM(256, return_sequences=True)(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.Model(inputs=inputA, outputs=x)
    # Second input will process the structural of the url
    y = tf.keras.layers.Dense(6, activation="relu")(inputB)
    y = tf.keras.Model(inputs=inputB, outputs=y)
    # Combine the processing of the text and structural of the url
    combined = tf.keras.layers.concatenate([x.output, y.output])
    # Apply softmax
    z = tf.keras.layers.Dense(1, activation='sigmoid')(combined)

    model = tf.keras.Model(inputs=[x.input, y.input], outputs=z)

    model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])

    # Train the model
    history = model.fit(x=[X_train_nlp, X_train_features], y=Y_train, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE,
                        workers=8, use_multiprocessing=True,
                        class_weight=CLASS_WEIGHTS)

    return model

def model_eval(model, X_test_nlp, X_test_features, Y_test):
    # Inferencing the test data
    Y_pred = model.predict([X_test_nlp, np.array(X_test_features)])
    X_test['pred'] = Y_pred
    X_test['label'] = Y_test
    # Plotting precision-recall curve
    recall = []
    precision = []
    ratio_malicious_benign = 0.05
    flag_pass = False
    thr_final = 0
    for thr in np.arange(0, 1, 0.01):
        FPs = len(X_test[(X_test['pred'] > thr) & (X_test['label'] == 0)])
        len_ben = len(X_test[X_test['label'] == 0])
        len_mal = len_ben * ratio_malicious_benign
        recall_step = len(X_test[(X_test['pred'] > thr) & (X_test['label'] == 1)]) / len(X_test[X_test['label'] == 1])
        recall.append(recall_step)
        TPs = len_mal * recall_step
        precision.append(TPs / (TPs + FPs))
        if TPs / (TPs + FPs) > 0.9 and flag_pass == False:
            print('Presicion: {}'.format(TPs / (TPs + FPs)))
            print('Recall: {}'.format(recall_step))
            print('Threshhold: {}'.format(thr))
            thr_final = thr
            flag_pass = True
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('URLS model')

def save_model(df_max_min, tokenizer_df, model):
    df_max_min.to_csv('max_min_urls.csv', index=False)
    tokenizer_df.to_csv('tokenizer_urls.csv', index=False)
    model.save('url_model_keras')

# Read Alexa rank domain dataframe
alexa_rank = pd.read_csv('../datasets/alexa-top-500k.csv',header=None)
alexa_rank.columns = ['index','url']
alexa_rank_domain = alexa_rank['url'].apply(get_domain)
alexa_rank_1k = alexa_rank_domain.iloc[0:1000]
alexa_rank_100k = alexa_rank_domain.iloc[1000:100000]

alexa_rank_1k_domain_unique = pd.unique(alexa_rank_1k)
alexa_rank_100k_domain_unique = pd.unique(alexa_rank_100k)

url_df = pd.read_csv("../datasets/url_dataset.csv")

print("Processing data for url model...")
X_train, Y_train, X_test, Y_test = data_preprocessing(url_df)

print("Calculating stractural URL features...")
X_train_features, X_test_features, df_max_min = stractural_processing(X_train, X_test)

print("Calculating NLP URL features...")
X_train_nlp, X_test_nlp, tokenizer_df, vocab_length = nlp_processing(X_train, X_test)

print("Train URL model...")
model = train_model(X_train_nlp, X_train_features, Y_train)

print("Evaluate URL model...")
model_eval(model, X_test_nlp, X_test_features, Y_test)