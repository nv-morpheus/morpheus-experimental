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

import pandas as pd
import numpy as np
import matplotlib.pyplot as pyplot
from sklearn.preprocessing import LabelEncoder

import tldextract

import torch
from torch.utils.data import TensorDataset, DataLoader

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Consts
EPOCHS = 30
BATCH_SIZE = 1000
MALICIOUS_RATIO = 0.01
LEARNING_RATE = 0.001

TRAIN_BATCH_SIZE = 500
TEST_BATCH_SIZE = 500
EMB_SIZE = 10
EPOCHS_SIAMESE = 120
LEARNING_RATE_SIAMESE = 1e-3
CLASS_WEIGHTS = {0: 100, 1:1}
ALPHA = 1.5  # value between 0-2
DGA_FAMILIES = ['goz', 'bazarbackdoor', 'bamital', 'gspy', 'dyre', 'enviserv', 'chinad', 'monerodownloader',
                            'emotet', 'ramdo', 'padcrypt', 'qadars', 'banjori', 'corebot', 'rovnix', 'flubot',
                            'gameover', 'alexa']

def get_domain(url):
    domain = tldextract.extract(url).domain
    if domain == 'ddns':
        print(url)
        urls = url.split('.')
        urls_i = urls.index('ddns')
        if urls_i == 0:
            return 'ddns'
        print(urls[urls_i - 1])
        return urls[urls_i - 1]
    if domain:
        return domain
    return ''

def get_domain_space(domain):
    try:
        return " ".join(domain)
    except:
        print(domain)
        return ""

def split_train_test_dga(df, ratio=0.8):
    df_dga = df[df['label'] == 1]
    df_legit = df[df['label'] == 0]
    X_dga, y_dga = df_dga['domain_1'], df_dga['label']
    X_legit, y_legit = df_legit['domain_1'], df_legit['label']
    train_dga_i = []
    train_ben_i = []
    test_dga_i = []
    test_ben_i = []
    # Make the dga train set to be more equale between families without dominant family
    for fam in pd.unique(df_dga['type']):
        df_dga_fam = df_dga[df_dga['type'] == fam]
        # Shuffle the dataframe rows
        df_dga_fam = df_dga_fam.sample(frac=1)
        if len(df_dga_fam) > 10000:
            train_dga_i.extend(df_dga_fam.iloc[0:int(ratio * 10000)].index)
            test_dga_i.extend(df_dga_fam.iloc[int(ratio * 10000):].index)
        else:
            train_dga_i.extend(df_dga_fam.iloc[0:int(ratio * len(df_dga_fam))].index)
            test_dga_i.extend(df_dga_fam.iloc[int(ratio * len(df_dga_fam)):].index)
    df_legit = df_legit.sample(frac=1)
    train_ben_i.extend(df_legit.iloc[0:int(ratio * len(df_legit))].index)
    test_ben_i.extend(df_legit.iloc[int(ratio * len(df_legit)):].index)
    train_dga_i.extend(train_ben_i)
    test_dga_i.extend(test_ben_i)
    X_train = df['domain_1'][train_dga_i]
    y_train = df['label'][train_dga_i]
    X_test = df['domain_1'][test_dga_i]
    y_test = df['label'][test_dga_i]

    return X_train, X_test, y_train, y_test

def create_data_loader(data, label):
    tensor_data = torch.Tensor(data.astype(int))
    tensor_label = torch.Tensor(label)
    my_dataset = TensorDataset(tensor_data, tensor_label)
    data_loader = DataLoader(my_dataset, batch_size=BATCH_SIZE, shuffle=True)
    return data_loader

def create_batch_offline(indices, batch_size, anc_indices, domain_indices=None):
    """choose an anchor, a positive and a negative batch.
    if domain_indices is given, choose anchor only from the specified domains indices. """
    x_anchors = np.zeros((batch_size, 75))
    x_positives = np.zeros_like(x_anchors)
    x_negatives = np.zeros_like(x_anchors)

    y = encoded_labels[indices]
    anc_indices = np.intersect1d(anc_indices, domain_indices, assume_unique=True)
    for i in range(0, batch_size):
        anc_idx = np.random.choice(anc_indices)
        x_anchor = X_data[anc_idx]
        y_anchor = encoded_labels[anc_idx]

        indices_for_pos = indices[np.where(y == y_anchor)]  # resulting array alway >=1 (the anchor itself)
        pos_idx = np.random.choice(indices_for_pos)
        indices_for_neg = indices[np.where(y != y_anchor)]
        neg_idx = np.random.choice(indices_for_neg)

        x_positive = X_data[pos_idx]
        x_negative = X_data[neg_idx]

        x_anchors[i] = x_anchor
        x_positives[i] = x_positive
        x_negatives[i] = x_negative

    return [x_anchors, x_positives, x_negatives]

def triplets_generator(**kwargs):
    while True:
        x = create_batch_offline(**kwargs)
        dummy_y = np.zeros(
            (x[0].shape[0], 3, EMB_SIZE))  # dummy y (never used) the size of the siamese input is required
        yield x, dummy_y

def triplet_loss(y_true, y_pred):
    anchor, positive, negative = y_pred[:, :EMB_SIZE], y_pred[:, EMB_SIZE:2 * EMB_SIZE], y_pred[:, 2 * EMB_SIZE:]
    p_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
    n_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
    return tf.reduce_mean(tf.maximum(0., p_dist - n_dist + ALPHA))

def argmin_label(row, ref, ref_save):
    emb = np.array(row[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    list_dist = [np.sum(np.power(emb - ref_save[key], 2)) for key in ref]
    arg_m = np.argmin(list_dist)
    dist_m = np.min(list_dist)
    row['predict_label'] = list(domains)[arg_m]
    row['predict_dist'] = dist_m
    return row

def data_preprocessing_binary(df):
    # Make spaces between domain chars
    df['domain_1'] = df['domain'].apply(get_domain_space)
    # Split train-test
    X_train, X_test, y_train, y_test = split_train_test_dga(df, 0.8)

    domain_test = df['domain'].iloc[X_test.index]
    type_test = df['type'].iloc[X_test.index]
    # Convert text to tokens
    X_train_np = tokenizer.texts_to_sequences(X_train)
    X_train_np = pad_sequences(X_train_np, maxlen=75, padding='post')
    X_test_np = tokenizer.texts_to_sequences(X_test)
    X_test_np = pad_sequences(X_test_np, maxlen=75, padding='post')

    X_train = np.array(X_train_np).astype(int)
    X_test = np.array(X_test_np).astype(int)

    return X_train, y_train, X_test, y_test, domain_test, type_test

def train_model_binary(X_train, y_train, X_test, y_test):
    # Defining the model
    inputA = tf.keras.layers.Input(shape=(X_train.shape[1],), name='input')
    x = tf.keras.layers.Embedding(max_features, 128, input_length=75)(inputA)
    x = tf.keras.layers.Conv1D(filters=64, kernel_size=4, activation='relu')(x)
    x = tf.keras.layers.Conv1D(filters=64, kernel_size=4, activation='relu')(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(10, activation='relu')(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(x)
    x = tf.keras.Model(inputs=inputA, outputs=x)
    model = tf.keras.Model(inputs=x.input, outputs=x.output)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    batch_size = 1000
    model.fit(X_train, y_train, batch_size=batch_size, epochs=30,
              validation_data=([X_test], y_test), class_weight=CLASS_WEIGHTS)
    return model

def model_eval_binary(model, X_test, y_test, domain_test, type_test):
    y_pred = model.predict(X_test)
    D_test = pd.DataFrame()

    D_test["domain"] = domain_test
    D_test["type"] = type_test
    D_test["label"] = y_test
    D_test["pred"] = y_pred

    recall = []
    precision = []
    ratio_malicious_benign = 0.01
    flag_pass = False
    thr_final = 0
    for thr in np.arange(0, 1, 0.01):
        FPs = len(D_test[(D_test['pred'] > thr) & (D_test['label'] == 0)])
        len_mal = len(D_test[D_test['label'] == 0]) * ratio_malicious_benign
        recall_step = len(D_test[(D_test['pred'] > thr) & (D_test['label'] == 1)]) / len(D_test[D_test['label'] == 1])
        recall.append(recall_step)
        TPs = len_mal * recall_step
        precision.append(TPs / (TPs + FPs))
        if TPs / (TPs + FPs) > 0.9 and flag_pass == False:
            print('Precision: {}'.format(TPs / (TPs + FPs)))
            print('Recall: {}'.format(recall_step))
            print('Threshhold: {}'.format(thr))
            thr_final = thr
            flag_pass = True
    pyplot.plot(recall, precision, marker='.', label='CNN Pytorch')
    pyplot.xlabel('Recall')
    pyplot.ylabel('Precision')
    pyplot.title('DGA model')

    D_test_mal = pd.DataFrame(D_test[D_test['label'] == 1].groupby(['type'], as_index=False)['label'].sum())
    D_test_mal_detected = pd.DataFrame(
        D_test[(D_test['label'] == 1) & (D_test['pred'] > thr_final)].groupby(['type'], as_index=False)['label'].sum())
    D_test_mal_detected.columns = ['type', 'detected']
    D_test_mal = pd.merge(D_test_mal, D_test_mal_detected, how="left", on=["type"])
    D_test_mal['detected'] = D_test_mal['detected'].fillna(0)
    D_test_mal['ratio'] = D_test_mal['detected'] / D_test_mal['label']
    print(D_test_mal[(D_test_mal['ratio'] < thr_final) & (D_test_mal['label'] > D_test_mal['label'].median())])
    print(D_test_mal[(D_test_mal['ratio'] > thr_final) & (D_test_mal['label'] > D_test_mal['label'].median())])

def data_preprocessing_families(df):
    # Merge dgas families with the same pattern
    df['type'] = df['type'].replace('FluBot_dga', 'flubot')
    df['type'] = df['type'].replace('fobber_v2', 'fobber')
    df['type'] = df['type'].replace('legit', 'alexa')
    df['type'] = df['type'].replace('pykspa_v2_real', 'pykspa')
    df['type'] = df['type'].replace('pykspa_v2_fake', 'pykspa')
    df['type'] = df['type'].replace('gameoverdga', 'gameover')
    # Merge others dgas families to 1 label
    for type_dga in pd.unique(df['type']):
        if type_dga not in DGA_FAMILIES:
            df['type'] = df['type'].replace(type_dga, 'alexa')

    df.reset_index(drop=True, inplace=True)

    labels_type = df['type']
    # Label encoding the dga families
    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels_type)
    dict_type_count = df['type'].value_counts().to_dict()
    dict_type_count.pop('alexa')
    df['domain_1'] = df['domain'].apply(get_domain_space)

    X_train, X_test, y_train, y_test = split_train_test_dga(df, 0.8)

    # Convert text to tokens
    X_data = tokenizer.texts_to_sequences(df['domain_1'])
    X_data = pad_sequences(X_data, maxlen=75, padding='post')
    domains = dict_type_count.keys()
    df['new_col'] = df['type'].isin(domains).astype(int)
    domains_idx = np.array(df.index[df['new_col'] == 1])
    noise_idx = np.array(df.index[df['new_col'] == 0])
    indices = domains_idx
    train_indices_same = np.intersect1d(X_train.index, domains_idx, assume_unique=False)
    train_indices_diff = np.intersect1d(X_train.index, noise_idx, assume_unique=False)
    test_indices_same = np.intersect1d(X_test.index, domains_idx, assume_unique=False)
    test_indices_diff = np.intersect1d(X_test.index, noise_idx, assume_unique=False)
    train_classes, train_cnt = np.unique(encoded_labels[train_indices_same], return_counts=True)
    test_classes, test_cnt = np.unique(encoded_labels[test_indices_same], return_counts=True)
    stacked = np.stack((train_cnt, test_cnt), axis=1)

    anc_idx = np.random.choice(train_indices_same)
    anchor = X_data[anc_idx]
    encoded_labels_train = encoded_labels[domains_idx]
    steps_per_epoch = int(train_indices_same.size / TRAIN_BATCH_SIZE)
    validation_steps = int(test_indices_same.size / TEST_BATCH_SIZE)

    train_generator = triplets_generator(indices=X_train.index, batch_size=TRAIN_BATCH_SIZE,
                                         anc_indices=train_indices_same, domain_indices=train_indices_same)
    validation_generator = triplets_generator(indices=X_test.index, batch_size=TEST_BATCH_SIZE,
                                              anc_indices=test_indices_same, domain_indices=test_indices_same)

    return train_generator, validation_generator, X_data, encoded_labels, steps_per_epoch, domains, labels_type, train_indices_same, train_indices_diff, test_indices_same, test_indices_diff

def train_model_families(train_generator, steps_per_epoch):
    # Defining the model
    inputA = tf.keras.layers.Input(shape=(75,), name='input')
    x = tf.keras.layers.Embedding(max_features, 128, input_length=75)(inputA)
    x = tf.keras.layers.Conv1D(filters=64, kernel_size=4, activation='relu')(x)
    x = tf.keras.layers.Conv1D(filters=64, kernel_size=4, activation='relu')(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(EMB_SIZE, activation=None)(x)
    x = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name='output')(x)
    x = tf.keras.Model(inputs=inputA, outputs=x)
    model = tf.keras.Model(inputs=x.input, outputs=x.output)

    input_anchor = tf.keras.layers.Input(shape=(75))
    input_positive = tf.keras.layers.Input(shape=(75))
    input_negative = tf.keras.layers.Input(shape=(75))

    embedding_anchor = model(input_anchor)
    embedding_positive = model(input_positive)
    embedding_negative = model(input_negative)

    output = tf.keras.layers.concatenate([embedding_anchor, embedding_positive, embedding_negative], axis=1)

    siamese_net = tf.keras.models.Model([input_anchor, input_positive, input_negative], output)

    siamese_net.compile(loss=triplet_loss, optimizer=Adam(learning_rate=LEARNING_RATE_SIAMESE))

    history = siamese_net.fit(
        train_generator, steps_per_epoch=steps_per_epoch, epochs=EPOCHS_SIAMESE, workers=8, use_multiprocessing=True)

    return model

def model_eval_families(model, train_indices_same, train_indices_diff, test_indices_same, test_indices_diff):
    train_indices_same = np.sort(train_indices_same)
    train_indices_diff = np.sort(train_indices_diff)
    x_train_same = X_data[train_indices_same]
    x_train_diff = X_data[train_indices_diff]
    y_train_same = labels_type.iloc[train_indices_same]
    y_train_diff = labels_type.iloc[train_indices_diff]
    x_test_same = X_data[test_indices_same]
    x_test_diff = X_data[test_indices_diff]
    y_test_same = labels_type.iloc[test_indices_same]
    y_test_diff = labels_type.iloc[test_indices_diff]

    # x_train_raw = x_train.reshape(-1,32*32*3)
    x_train_same_emb = model.predict(x_train_same)
    x_train_diff_emb = model.predict(x_train_diff)

    # x_train_raw = x_train.reshape(-1,32*32*3)
    x_test_same_emb = model.predict(x_test_same)
    x_test_diff_emb = model.predict(x_test_diff)

    # Create a dict of the output model
    # create a dict of vector embeddings per class:
    ref = {}
    for domain in domains:
        print(domain)
        print(np.where(y_train_same == domain)[0])
        x_domain = x_train_same[np.where(y_train_same == domain)[0]]
        print(x_domain)
        ref[domain] = model(x_domain)
    ref_save = {}
    # Create dict of anchors
    for key in ref:
        ref_save[key] = ref[key][0]
    ref_save_df = pd.DataFrame()
    ref_save_df['Family'] = ref_save.keys()
    for i in range(len(ref_save['emotet'])):
        list_vec = []
        for key in ref_save:
            list_vec.append(ref_save[key][i].numpy())
        ref_save_df[i] = list_vec

    y_test_same_list = y_test_same.tolist()
    df_test_same_emb = pd.DataFrame(x_test_same_emb)
    df_test_same_emb['label'] = y_test_same_list
    df_test_same_emb_mini = df_test_same_emb
    df_test_same_emb_mini = df_test_same_emb_mini.swifter.apply(lambda row: argmin_label(row, ref, ref_save), axis=1)

    print(len(df_test_same_emb_mini[(df_test_same_emb_mini['predict_label'] == df_test_same_emb_mini['label']) & (
                df_test_same_emb_mini['predict_dist'] < 0.5)]) / len(df_test_same_emb_mini))

    y_test_diff_list = y_test_diff.tolist()
    df_test_diff_emb = pd.DataFrame(x_test_diff_emb)
    df_test_diff_emb['label'] = y_test_diff_list
    df_test_diff_emb_mini = df_test_diff_emb
    df_test_diff_emb_mini = df_test_diff_emb_mini.swifter.apply(lambda row: argmin_label(row, ref, ref_save), axis=1)

    print(len(df_test_diff_emb_mini[(df_test_diff_emb_mini['predict_dist']<0.5) & ~df_test_diff_emb_mini['predict_label'].isin(['simda','fobber','pykspa_v1'])])/len(df_test_same_emb_mini[(~df_test_same_emb_mini['label'].isin(['simda','fobber','pykspa_v1'])) & (df_test_same_emb_mini['predict_label']==df_test_same_emb_mini['label']) & (df_test_same_emb_mini['predict_dist']<0.5)]))

tokenizer = Tokenizer()
tokenizer.word_index = pd.read_csv('tokenizer.csv').set_index('keys')['values'].to_dict()
max_features = len(tokenizer.word_index) + 1

print("Reading tokenizer...")
# Read tokenizer
print("Reading data for training")
df_binary = pd.read_csv("../datasets/dga_training_dataset.csv")
df_families = df_binary.copy()

print("Processing data for binary model...")
X_train, y_train, X_test, y_test, domain_test, type_test = data_preprocessing_binary(df_binary)

print("Training binary model...")
model_binary = train_model_binary(X_train, y_train, X_test, y_test)

print("Saving binary model...")
model_binary.save('../dga_binary_keras_model')

print("Evaluating binary model...")
model_eval_binary(model_binary, X_test, y_test, domain_test, type_test)

print("Processing data for families model...")
train_generator, validation_generator, X_data, encoded_labels, steps_per_epoch, domains, labels_type, train_indices_same, train_indices_diff, test_indices_same, test_indices_diff = data_preprocessing_families(df_families)
print("Training families model...")
model_families = train_model_families(train_generator, steps_per_epoch)

print("Saving family model...")
model_families.save('../dga_family_keras_model')

print("Evaluating families model...")
model_eval_families(model_families, train_indices_same, train_indices_diff, test_indices_same, test_indices_diff)


