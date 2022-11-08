# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
import os


def get_metrics(pred, labels, out_dir, name='RGCN'):
    """Compute evaluation metrics

    Args:
        pred : prediction
        labels (_type_): groundtruth label
        out_dir (_type_): directory for saving
        name (str, optional): model name. Defaults to 'RGCN'.

    Returns:
        _type_: _description_
    """

    labels, pred, pred_proba = labels, pred.argmax(1), pred[:, 1]

    acc = ((pred == labels)).sum() / len(pred)

    true_pos = (np.where(pred == 1, 1, 0) + np.where(labels == 1, 1, 0) > 1).sum()
    false_pos = (np.where(pred == 1, 1, 0) + np.where(labels == 0, 1, 0) > 1).sum()
    false_neg = (np.where(pred == 0, 1, 0) + np.where(labels == 1, 1, 0) > 1).sum()
    true_neg = (np.where(pred == 0, 1, 0) + np.where(labels == 0, 1, 0) > 1).sum()

    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0

    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    confusion_matrix = pd.DataFrame(np.array([[true_pos, false_pos], [false_neg, true_neg]]),
                                    columns=["labels positive", "labels negative"],
                                    index=["predicted positive", "predicted negative"])

    ap = average_precision_score(labels, pred_proba)

    fpr, tpr, _ = roc_curve(labels, pred_proba)
    prc, rec, _ = precision_recall_curve(labels, pred_proba)
    roc_auc = auc(fpr, tpr)
    pr_auc = auc(rec, prc)

    # uncomment to save individual plots.
    # save_roc_curve(fpr, tpr, roc_auc, os.path.join(out_dir, name + "_roc_curve.png"), model_name=name)
    # save_pr_curve( rec, prc, pr_auc, ap, os.path.join(out_dir, name + "_pr_curve.png"), model_name=name)
    auc_r = (fpr, tpr, roc_auc, name)
    return acc, f1, precision, recall, roc_auc, pr_auc, ap, confusion_matrix, auc_r


def save_roc_curve(fpr, tpr, roc_auc, location, model_name='Model'):
    """Produce and save AUC ROC curve

    Args:
        fpr (_type_): false positive rate
        tpr (_type_): true negative rate
        roc_auc (_type_): auc score
        location (_type_): location dir
        model_name (str, optional): model name. Defaults to 'Model'.
    """
    f = plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(model_name + '  ROC curve')
    plt.legend(loc="lower right")
    f.savefig(location)


def save_pr_curve(rec, prec, pr_auc, ap, location, model_name='Model'):
    """Produce and save precision-recall curve

    Args:
        rec (_type_): recall
        prec (_type_): precision
        pr_auc (_type_): precision-recall curve
        ap (_type_): average precision
        location (_type_): location dir
        model_name (str, optional): model name. Defaults to 'Model'.
    """

    f = plt.figure()
    lw = 2
    plt.plot(rec, prec, color='darkorange', lw=lw, label='PR curve (area = %0.2f)' % pr_auc)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(model_name + ' PR curve: AP={0:0.2f}'.format(ap))
    plt.legend(loc="lower right")
    f.savefig(location)


def precision_top_k_day(df_day, top_k, model_name, user_id='userId_id'):
    """ This takes the max of the predictions AND the max of label FRAUD for each User_ID,
    and sorts by decreasing order of fraudulent prediction

    Args:
        df_day (dataframe): dataframe of scores
        top_k (int): K in top k
        model_name (string): model name
        user_id (str, optional): user_id column. Defaults to 'userId_id'.

    Returns:
        _type_: precision @topk of given day.
    """

    df_day = df_day.groupby(user_id).max().sort_values(by=model_name, ascending=False).reset_index(drop=False)

    # Get the top k most suspicious users
    df_day_top_k = df_day.head(top_k)
    list_detected_compromised_users = list(df_day_top_k[df_day_top_k.fraud_label == 1][user_id])

    # Compute precision top k
    user_precision_top_k = len(list_detected_compromised_users) / top_k

    return list_detected_compromised_users, user_precision_top_k


def user_precision_top_k(predictions_df, top_k, model_name, user_id='userId_id'):
    """ Precision at top K for user at given day.

    Args:
        predictions_df (_type_): prediction score.
        top_k (_type_): k rank
        model_name (_type_): model used
        user_id (str, optional): UserId. Defaults to 'userId_id'.

    Returns:
        _type_: precision top k/day
    """

    # Sort days by increasing order
    list_days = list(predictions_df['day'].unique())
    list_days.sort()

    user_precision_top_k_per_day_list = []
    nb_compromised_users_per_day = []
    detected_users_at = {}
    # For each day, compute precision top k
    for day in list_days:

        df_day = predictions_df[predictions_df['day'] == day]
        df_day = df_day[[model_name, user_id, 'fraud_label']]

        nb_compromised_users_per_day.append(len(df_day[df_day.fraud_label == 1][user_id].unique()))

        detected_users, user_precision_top_k = precision_top_k_day(df_day, top_k, model_name)

        user_precision_top_k_per_day_list.append(user_precision_top_k)
        detected_users_at[day] = {'users': detected_users, 'prec': user_precision_top_k}

    # Compute the mean
    mean_user_precision_top_k = np.array(user_precision_top_k_per_day_list).mean()

    # Returns precision top k per day as a list, and resulting mean
    return nb_compromised_users_per_day, user_precision_top_k_per_day_list, mean_user_precision_top_k, detected_users_at


#  Baseline models for comparison.


def baseline_models(train_x, test_x, train_idx, test_idx, labels, test_label, name='XGB', result_dir='azure_result'):
    """This trains and produce metric from supervised baseline model (XGBoost)

    Args:
        train_x : training data
        test_x : test data
        train_idx : training index
        test_idx : test index
        labels : groundtruth label
        test_label : evalaution label
        name (str, optional): model name. Defaults to 'XGB'.
        result_dir (str, optional): result directory. Defaults to 'azure_result'.

    Returns:
        list: evaluation metrics such acc, f1, precision, recall, roc_auc, pr_auc
    """

    from xgboost import XGBClassifier

    classifier = XGBClassifier(n_estimators=100)

    # baseline_train = train_data.drop(col_remove, axis=1)
    classifier.fit(train_x, labels[train_idx])
    baseline_pred = classifier.predict_proba(test_x)

    # compute metrics on prediction
    acc, f1, precision, recall, roc_auc, pr_auc, ap, confusion_matrix, roc_r = get_metrics(
        baseline_pred, test_label[test_idx], out_dir=result_dir, name=name, )
    return acc, f1, precision, recall, roc_auc, pr_auc, ap, confusion_matrix, baseline_pred[:, 1], roc_r


def unsupervised_models(train_x,
                        test_x,
                        train_idx,
                        test_idx,
                        labels,
                        test_label,
                        name='iforest',
                        result_dir='azure_result'):
    """This trains and produce metric from unsupervised baseline model (iforest)

    Args:
        train_x (pd.DataFrame): train data
        test_x (pd.DataFrame): test data
        train_idx (list): training data index
        test_idx (list): test data index
        labels (list): training label
        test_label (list): test labe
        name (str, optional): model name. Defaults to 'iforest'.
        result_dir (str, optional): metrics

    Returns:
        list : evaluation metrics such acc, f1, precision, recall, roc_auc, pr_auc
    """
    from sklearn.ensemble import IsolationForest
    ff = IsolationForest(n_estimators=100)
    ff.fit(train_x)
    baseline_pred = -ff.score_samples(test_x)

    baseline_pred = np.hstack(((1 - baseline_pred).reshape(-1, 1), baseline_pred.reshape(-1, 1)))

    acc, f1, precision, recall, roc_auc, pr_auc, ap, confusion_matrix, roc_r = get_metrics(
        baseline_pred, test_label[test_idx], out_dir=result_dir, name=name, )
    return acc, f1, precision, recall, roc_auc, pr_auc, ap, confusion_matrix, baseline_pred[:, 1], roc_r

def compare_models(test_logits, fraud_labels, test_seed, train_seed, train_embed, test_embed, outdir='./'):
    # RGCN
    acc, f1, precision, recall, roc_auc, pr_auc, ap, confusion_matrix, _ = get_metrics(
        test_logits.numpy(), fraud_labels[test_seeds], out_dir=result_dir, name='RGCN')


    test_scores['rgcn'] = test_logits.numpy()[:,1]

    # XGBoost
    acc, f1, precision, recall, roc_auc, pr_auc, ap, confusion_matrix, pred_xgb, roc_xgg = baseline_models(
        train_x=train_data.drop(col_remove, axis=1), test_x=test_data.drop(col_remove, axis=1),
        train_idx=train_idx, test_idx=test_idx, labels=labels.cpu().numpy() ,
        test_label=fraud_labels, name='XGB')

    test_scores['xgb'] = pred_xgb

    # RGCN+XGBoost
    acc, f1, precision, recall, roc_auc, pr_auc, ap, confusion_matrix, pred_rgcn_xgb, _ = baseline_models(
        train_x=train_embedding.cpu().numpy(),
        test_x=test_embedding.cpu().numpy(), 
        train_idx=train_seeds,
        test_idx=test_seeds,
        labels=labels.cpu().numpy(),
        test_label=fraud_labels,
        name='RGCN+XGB')

    test_scores['rgcn_xgb'] = pred_rgcn_xgb

    # rgcn + iforest
    acc, f1, precision, recall, roc_auc, pr_auc, ap, confusion_matrix, pred_rgcn_if, roc_rgcniff = unsupervised_models(
        train_x=train_embedding.cpu().numpy(), test_x=test_embedding.cpu().numpy(), 
        train_idx=train_seeds, test_idx=test_seeds,
        labels=labels.cpu().numpy(),
        test_label=fraud_labels, name='RGCN_iforest')

    test_scores['rgcn_if'] = pred_rgcn_if

    #iforest
    acc, f1, precision, recall, roc_auc, pr_auc, ap, confusion_matrix, iff_scores,_ = unsupervised_models(
        train_x=train_data.drop(col_remove, axis=1), test_x=test_data.drop(col_remove, axis=1), 
        train_idx=train_data.index, test_idx=test_data.reset_index().index,
        labels=test_data.fraud_label,
        test_label=fraud_labels, name='iforest')