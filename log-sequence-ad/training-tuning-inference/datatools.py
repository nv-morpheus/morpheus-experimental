# SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#
# Description : This file implements portion of negative sampling & sliding windows
# Author      : Copyright (c) 2021 Xiao Han
# License     : MIT

import random
import warnings
from collections import deque
from itertools import islice

import numpy as np
import pandas as pd
import torch
from gensim.models import Word2Vec

warnings.filterwarnings("ignore")


def ngrams(sequence, n, **kwargs):
    """compute ngram. This method is based on nltk.util.ngrams implementation.

    Parameters
    ----------
    sequence : list
        List of strings
    n : int
        ngram param. set n=2 for bigram

    Yields
    ------
    _type_
        _description_
    """

    # sliding_window('ABCDEFG', 4) --> ABCD BCDE CDEF DEFG
    # https://docs.python.org/3/library/itertools.html?highlight=sliding_window#itertools-recipes
    it = iter(sequence)
    window = deque(islice(it, n), maxlen=n)
    if len(window) == n:
        yield tuple(window)
    for x in it:
        window.append(x)
        yield tuple(window)


def bigrams(sequence, **kwargs):
    yield from ngrams(sequence, 2, **kwargs)


def preprocess(df, window_size=100, step_size=20):
    '''Preprocessing structured log dataset

    Args:
        df: dataframe of structured log dataset
        window_size: length of sliding window
        step_size: step length

    Return:
        DataFrame of preprocessed sliding windows
    '''
    df["Label"] = df["Label"].apply(lambda x: int(x != '-'))
    df = df[["Label", "EventId"]]
    df["Key_label"] = df["Label"]
    log_size = df.shape[0]
    label_data = df.iloc[:, 0]
    logkey_data = df.iloc[:, 1]
    new_data = []
    index = 0
    while index <= log_size - window_size:
        new_data.append([
            max(label_data[index:index + window_size]),
            logkey_data[index:index + window_size].values,
            label_data[index:index + window_size].values
        ])
        index += step_size
    return pd.DataFrame(new_data, columns=df.columns)


def get_dataframe(lst, label, dic):
    df = pd.DataFrame()
    df['EventId'] = lst
    df['class_label'] = label
    return str_key_to_w2v_index(df, dic)


def get_training_dictionary(df):
    '''Get training dictionary

    Arg:
        df: dataframe of preprocessed sliding windows

    Return:
        dictionary of training data
    '''
    dic = {}
    count = 0
    for i in range(len(df)):
        lst = list(df['EventId'].iloc[i])
        for j in lst:
            if j in dic:
                pass
            else:
                dic[j] = str(count)
                count += 1
    return dic


def str_to_str_keys(df, dic):
    '''Convert original parser log keys into number version of log keys

    Args:
        df: dataframe which needs to be converted
        dic: reference dictionary

    Return:
        df: dataframe which EventId column has been converted
    '''
    for i in range(len(df)):
        lst = list(df['EventId'].iloc[i])
        temp = []
        for j in lst:
            if j in dic:
                temp.append(dic[j])
            else:
                temp.append(str(len(dic)))
        df['EventId'].iloc[i] = temp
    return df


def get_bigram(df):
    '''Get the bigram according to the input dataframe

    Arg:
        df: pd.dataframe
        dataframe used to compute bigrams

    Returns:
        bigram: dictionary of bigrams
        uni: sliding window first log key
    '''
    bigram = {}
    uni = []
    for i in range(len(df)):
        temp_lst = list(df['EventId'].iloc[i])
        if temp_lst[0] not in uni:
            uni.append(temp_lst[0])
        for a, b in bigrams(temp_lst):
            a = str(a)
            b = str(b)
            if a in bigram:
                if b not in bigram.get(a):
                    bigram[a].append(b)
            else:
                bigram[a] = [b]
    return bigram, uni


def get_w2v_dic(w2v):
    '''Get Word2Vec dictionary

    Arg:
        w2v: Word2Vec model

    Return:
        dic: dictionary of Word2Vec model
    '''
    return {i: w2v.wv.vocab.get(i).index for i in list(w2v.wv.vocab)}


def str_key_to_w2v_index(df, dic):
    '''Chenge string number keys into Word2Vec int keys

    Args:
        df: DataFrame of data which needs to be changed for InterpretableSAD model
        dic: reference dictionary

    Return:
        df: DataFrame of modified data
    '''
    lst_w2v = []
    for i in range(len(df)):
        lst = list(df['EventId'].iloc[i])
        temp = []
        for j in lst:
            if j in dic:
                temp.append(dic[j])
            else:
                print('Error: key is not in the dict')
        lst_w2v.append(temp)
    df['W2V_EventId'] = lst_w2v
    return df


def sliding_window(dataset_name, window_size=100, step_size=20, train_size=100000):
    '''Cut log data into sliding windows and train a Word2Vec model

    Args:
        dataset_name: name of log dataset
        window_size: length of sliding window
        step_size: length of step
        train_size: number of training samples

    Returns:
        train_normal: DataFrame of training normal samples
        test_normal: DataFrame of testing normal samples
        test_abnormal: DataFrame of testing abnormal samples
        bigram: dictionary of bigrams from training data
        unique: list of start log keys of training data
        weights: weight matrix of Word2Vec
        train_dict: dictionary of training data
        w2v_dic: dictionary of Word2Vec
    '''
    print('Reading: ' + dataset_name)
    df = pd.read_csv(dataset_name)

    print('Total logs in the dataset: ', len(df))
    window_df = preprocess(df, window_size, step_size)
    df_normal = window_df[window_df["Label"] == 0]

    # shuffle normal data
    df_normal = df_normal.sample(frac=1, random_state=42).reset_index(drop=True)
    normal_len = len(df_normal)
    train_normal = df_normal[:train_size]
    print("training size {}".format(train_size))

    test_normal = df_normal[train_size:]
    print("test normal size {}".format(normal_len - train_size))

    test_abnormal = window_df[window_df["Label"] == 1]
    print('test abnormal size {}'.format(len(test_abnormal)))

    # get dictionary of training data and total data
    train_dict = get_training_dictionary(train_normal)
    print('Number of training keys:', len(train_dict))

    # change the original log keys into number log keys based on the training dictionary
    train_normal = str_to_str_keys(train_normal, train_dict)

    # get the bigram dictionary and unique list from the training data
    bigram, unique = get_bigram(train_normal)

    # define training data
    sentences = list(train_normal.EventId.values)
    sentences.append([str(len(train_dict))])

    # train model
    w2v = Word2Vec(sentences, size=8, min_count=1, seed=1)
    # summarize the loaded model
    print('Word2Vec model:', w2v)
    # get the Word2Vec model weights for lstm embedding layer
    weights = torch.FloatTensor(w2v.wv.vectors)
    # get the Word2Vec dictionary
    w2v_dic = get_w2v_dic(w2v)

    # change the data with Word2Vec dictionary
    train_normal = str_key_to_w2v_index(train_normal, w2v_dic)

    # train_normal = test_vector(train_normal, train_dict, w2v_dic)
    test_normal = test_vector(test_normal, train_dict, w2v_dic)
    test_abnormal = test_vector(test_abnormal, train_dict, w2v_dic)

    return train_normal, test_normal, test_abnormal, bigram, unique, weights, train_dict, w2v_dic


def test_vector(test_normal_df, train_dict, w2v_dic):
    # change the original log keys into number log keys based on the training dictionary
    test_normal = str_to_str_keys(test_normal_df, train_dict)
    # change the data with Word2Vec dictionary
    test_normal = str_key_to_w2v_index(test_normal, w2v_dic)
    return test_normal


def get_neg_samp(window, index, bigram, uni, vocab_dim):
    '''Get negative sample for each given sliding window

    Args:
        window: a sample of normal log sequence
        index: list of indexes which need to be replaced
        bigram: bigram dictionary for reference
        uni: list of start log keys for reference
        vocab_dim: int of vocabulary size

    Return:
        window: a negative sample
    '''
    for i in index:
        if i == 0:
            in_bag = set(uni)
            out_bag = set(range(0, vocab_dim)).difference(in_bag)
            window[i] = str(random.sample(out_bag, 1)[0])
        else:
            if str(window[i]) in bigram:
                in_bag = set(bigram.get(str(window[i])))
                out_bag = set(range(0, vocab_dim)).difference(in_bag)
                window[i + 1] = str(random.sample(out_bag, 1)[0])
            else:
                out_bag = set(range(0, vocab_dim))
                window[i + 1] = str(random.sample(out_bag, 1)[0])
    return window


def negative_sampling(dataset, bigram, uni, number_times, vocab_dim):
    '''Negative sampling method

    Args:
        dataset: DataFrame of training normal dataset
        bigram: bigram dictionary for reference
        uni: list of start log keys for reference
        number_times: int number of times to decide the size of negative sampling size
        vocab_dim: int of vocabulary size

    Return:
        list of negative samples
    '''
    length = len(dataset)
    re_len = int(number_times * length)
    re_list = []
    lst_keys = list(dataset['EventId'])
    samples = list(np.random.random_integers(length - 1, size=(re_len, )))
    for i in map(lst_keys.__getitem__, samples):
        replace_n = random.randint(1, len(i))
        rep_index = random.choices(range(len(i) - 1), k=replace_n)
        temp = i[:]
        while temp in lst_keys:
            temp = get_neg_samp(temp, rep_index, bigram, uni, vocab_dim)
        re_list.append(temp)
    return re_list
