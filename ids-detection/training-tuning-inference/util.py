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

from itertools import permutations

import cupy as cp
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

import cudf
from cuml.decomposition import PCA
from cuml.metrics import precision_recall_curve
from cuml.metrics import roc_auc_score


def average_precision_score(y_true, y_score):
    """
    Compute average precision score using precision and recall computed from cuml.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    # return step function integral
    return -cp.sum(cp.diff(recall) * cp.array(precision)[:-1])


def metrics(y_true, y_score):
    """ AUC and AP scores.

    Parameters
    ----------
    y_true : series
        label
    y_score : series
        predicted scores

    Returns
    -------
    tuple
        auc and ap scores
    """
    auc = roc_auc_score(y_true=y_true, y_score=y_score)
    ap = average_precision_score(y_true, y_score)
    return [auc, ap]


def plot_roc(label, y_scores):
    """Plot graph of roc curve
    """
    fpr, tpr, _ = roc_curve(y_true=label.values.tolist(), y_score=y_scores.tolist())
    auc = metrics(label, y_scores)[0]
    plt.plot(fpr, tpr, label="ROC = " + str(np.round(auc, 2)))
    plt.plot(np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1), 'r-')
    plt.ylabel('tpr')
    plt.xlabel('fpr')
    plt.legend(loc='best')
    plt.title('Area under AUC curve')


def plot_pr(label, y_scores):
    """Plot graph of AP curve
    """
    ap = metrics(label, y_scores)[1]
    precision, recall, _ = precision_recall_curve(label, y_scores)
    plt.plot(recall, precision, label='AP = ' + str(np.round(ap, 2)))
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.legend(loc='best')
    plt.title('Area under PR curve')


def missing_values_table(df):
    """Generate missing values ratio of columns

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe

    Returns
    -------
    pd.DataFrame
       missing value ratio of columns
    """
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(columns={0: 'Missing Values', 1: '% of Total Values'})
    mis_val_table_ren_columns = mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
    print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
          "There are " + str(mis_val_table_ren_columns.shape[0]) + " columns that have missing values.")
    return mis_val_table_ren_columns


def load_dataset(dir_path):
    """Load dataset using cudf

    Parameters
    ----------
    dir_path : str
        file path name

    Returns
    -------
    cudf.DataFrame
        column formatted dataframe.
    """
    df = cudf.read_csv(dir_path)
    df.columns = ['_'.join(col.split()) for col in df.columns.str.strip()]
    return df


def drop_nonunique_features(df, nunique=1):
    """Drop nonunique columns

    Parameters
    ----------
    df : cudf.DataFrame
        _description_
    nunique : int, optional
        number of unique values, by default 1

    Returns
    -------
    cudf.DataFrame
        new cudf Dataframe
    """

    var_col = []
    for i in df.columns:
        if df[i].value_counts().nunique() <= nunique:
            var_col.append(i)
    return df.drop(var_col, axis=1, inplace=False)


def remove_naninf(df):
    """Remove all missing rows with inf, nan
    """
    temp_df = df.select_dtypes('float64')
    na_rows = temp_df.isin([cp.inf, cp.nan, -cp.inf]).any(1)
    return df[~na_rows]


def remove_categorical_features(df):
    """
    Remove categorical features from the cic dataset.
    """
    df = df.select_dtypes(['float64', 'int64'])
    return df.drop(["Protocol", "Source_Port", "Destination_Port"], axis=1, inplace=False)


def remove_correlated_features(df):
    """
    Remove features that are highly correlated. Mostly remove derived features.
    param df: pd.DataFrame
    """
    corr_mat = df.corr()
    # Select upper triangle of correlation matrix
    upper = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(np.bool))
    # Find features with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    return to_drop, df.drop(to_drop, inplace=False, axis=1)


def entropy(series):
    """Compute entropy of series values.

    Parameters
    ----------
    series : pd.Series
       Series numeric number.

    Returns
    -------
    computed entropy
    """
    vc = series.value_counts(normalize=True, sort=False)
    return -(vc * np.log(vc) / np.log(series.shape[0])).sum()


class NetFlowFeatureProcessing:

    def __init__(self, input_name, config=None) -> None:
        """Process raw netflow and perform transformation and feature engineering.

        Parameters
        ----------
        input_name : str
            input file name
        config : json, optional
            training configuration, by default None. If None perform
            training otherwise set inference model to true
        """

        self.input_name = input_name
        self.pca_component = None
        self.config = config
        if config:
            self.inference_mode = True
        else:
            self.inference_mode = False
            self.config = {}
            self.config['apply_pca'] = False

    def exract_features(self, df):
        """Extract features

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe input

        Returns
        -------
        pd.DataFrame
            Cleaned dataframe
        """

        if self.inference_mode:
            selected_columns = self.config['training_columns']
            df = df[selected_columns]
        else:
            df = df.pipe(drop_nonunique_features)

            df = remove_categorical_features(df)
            _, df = remove_correlated_features(df)
            self.config['training_columns'] = df.columns.tolist()
        return df

    def _encode_entropy(self, df, feature, target):
        """Perform pairwise feature entropy

        Parameters
        ----------
        df : pd.DataFrame

        feature : str
            feature column
        target : str
            target column feature

        Returns
        -------
        pd.DataFrame
            transformed features
        """

        feature_entropy = df.groupby(feature).agg(**{target + "_" + feature: (target, entropy)})
        return df.join(feature_entropy, on=feature, rsuffix="r_")

    def _transform(self, X, variance=.99):
        """Perform PCA transformation

        Parameters
        ----------
        X : cudf.ndarray
            input ndarray frame
        variance : float, optional
            pca variance to preserve, by default .99

        Returns
        -------
        _type_
            _description_
        """
        # transform using PCA
        if self.inference_mode:
            # for inference use number of components from config
            n_components = int(self.config['n_pca_components'])
            X = PCA(n_components=n_components, whiten=False).fit_transform(X)
        else:
            # training mode
            pca = PCA(n_components=X.shape[1])
            pca_xtrain = pca.fit_transform(X)
            variance_exp = cp.cumsum(pca.explained_variance_ratio_)
            n_components = variance_exp <= variance
            self.config['apply_pca'] = True
            self.config['n_pca_components'] = int(sum(n_components))
            self.config['pca_variance'] = variance
            X = pca_xtrain[:, n_components]
        return X

    def process(self, apply_pca=True):
        """Perform step by step extraction, transformation of Dataframe

        Parameters
        ----------
        apply_pca : bool, optional
            if true apply PCA, by default True

        Returns
        -------
        cudf.DataFrame
            final processed dataframe
        """

        df = load_dataset(self.input_name).pipe(remove_naninf).to_pandas()
        # Remove features
        cat_features = ['Source_IP', 'Destination_IP', 'Source_Port', 'Destination_Port']

        # Encode based on entropy of pairwise of categorical features.
        for feature, target in permutations(cat_features, 2):
            print(feature, target)
            df = self._encode_entropy(df, feature, target)

        df = self.exract_features(df)

        # Transform back from pandas to cudf to work with Loda.
        X_train = cudf.from_pandas(df).values

        # PCA transform
        if apply_pca:
            X_train = self._transform(X_train)

        return X_train, self.config
