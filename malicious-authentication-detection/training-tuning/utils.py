import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import logging


def get_logger(name):

    logger = logging.getLogger(name)
    log_format = '%(asctime)s %(levelname)s %(name)s: %(message)s'
    logging.basicConfig(format=log_format, level=logging.INFO,
                        filename='log.txt', filemode='w')
    logger.setLevel(logging.INFO)
    return logger


def get_metrics(pred, labels, out_dir, name='RGCN'):

    labels, pred, pred_proba = labels, pred.argmax(1), pred[:, 1]

    acc = ((pred == labels)).sum() / len(pred)

    true_pos = (np.where(pred == 1, 1, 0) + np.where(labels == 1, 1, 0) > 1).sum()
    false_pos = (np.where(pred == 1, 1, 0) + np.where(labels == 0, 1, 0) > 1).sum()
    false_neg = (np.where(pred == 0, 1, 0) + np.where(labels == 1, 1, 0) > 1).sum()
    true_neg = (np.where(pred == 0, 1, 0) + np.where(labels == 0, 1, 0) > 1).sum()

    precision = true_pos/(true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    recall = true_pos/(true_pos + false_neg) if (true_pos + false_neg) > 0 else 0

    f1 = 2*(precision*recall)/(precision + recall) if (precision + recall) > 0 else 0
    confusion_matrix = pd.DataFrame(
        np.array([[true_pos, false_pos],
                  [false_neg, true_neg]]),
        columns=["labels positive", "labels negative"],
        index=["predicted positive", "predicted negative"])

    ap = average_precision_score(labels, pred_proba)

    fpr, tpr, _ = roc_curve(labels, pred_proba)
    prc, rec, _ = precision_recall_curve(labels, pred_proba)
    roc_auc = auc(fpr, tpr)
    pr_auc = auc(rec, prc)

    save_roc_curve(fpr, tpr, roc_auc, os.path.join(
        out_dir, name+"_roc_curve.png"), model_name=name)
    #save_pr_curve( rec, prc, pr_auc, ap, os.path.join(out_dir, name + "_pr_curve.png"), model_name=name)
    auc_r = (fpr, tpr, roc_auc, name)
    return acc, f1, precision, recall, roc_auc, pr_auc, ap, confusion_matrix, auc_r


def save_roc_curve(fpr, tpr, roc_auc, location, model_name='Model'):
    f = plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(model_name + '  ROC curve')
    plt.legend(loc="lower right")
    f.savefig(location)


def save_pr_curve(rec, prc, pr_auc, ap, location, model_name='Model'):

    f = plt.figure()
    lw = 2
    plt.plot(rec, prc, color='darkorange',
             lw=lw, label='PR curve (area = %0.2f)' % pr_auc)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(model_name + ' PR curve: AP={0:0.2f}'.format(ap))
    plt.legend(loc="lower right")
    f.savefig(location)


def precision_top_k_day(df_day, top_k, model_name, user_id='userId_id'):

    # This takes the max of the predictions AND the max of label FRAUD for each User_ID,
    # and sorts by decreasing order of fraudulent prediction
    df_day = df_day.groupby(user_id).max().sort_values(
        by=model_name, ascending=False).reset_index(drop=False)

    # Get the top k most suspicious users
    df_day_top_k = df_day.head(top_k)
    list_detected_compromised_users = list(
        df_day_top_k[df_day_top_k.fraud_label == 1][user_id])

    # Compute precision top k
    user_precision_top_k = len(list_detected_compromised_users) / top_k

    return list_detected_compromised_users, user_precision_top_k


def user_precision_top_k(
        predictions_df, top_k, model_name, user_id='userId_id'):

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

        nb_compromised_users_per_day.append(
            len(df_day[df_day.fraud_label == 1][user_id].unique()))

        detected_users, user_precision_top_k = precision_top_k_day(
            df_day, top_k, model_name)

        user_precision_top_k_per_day_list.append(user_precision_top_k)
        detected_users_at[day] = {
            'users': detected_users, 'prec': user_precision_top_k}

    # Compute the mean
    mean_user_precision_top_k = np.array(
        user_precision_top_k_per_day_list).mean()

    # Returns precision top k per day as a list, and resulting mean
    return nb_compromised_users_per_day, user_precision_top_k_per_day_list, mean_user_precision_top_k, detected_users_at

#  Baseline comparison


def baseline_models(
        train_x, test_x, train_idx, test_idx, labels, test_label, name='XGB',
        result_dir='azure_result'):
    # , scale_pos_weight=sum(train_data.fraud_label==1) / sum(train_data.fraud_label==0))
    from xgboost import XGBClassifier

    classifier = XGBClassifier(n_estimators=100)

    #baseline_train = train_data.drop(col_remove, axis=1)
    classifier.fit(train_x, labels[train_idx])
    baseline_pred = classifier.predict_proba(test_x)

    acc, f1, precision, recall, roc_auc, pr_auc, ap, confusion_matrix, roc_r = get_metrics(
        baseline_pred, test_label[test_idx], out_dir=result_dir, name=name, )
    return acc, f1, precision, recall, roc_auc, pr_auc, ap, confusion_matrix, baseline_pred[:, 1], roc_r


def unsupervised_models(
        train_x, test_x, train_idx, test_idx, labels, test_label,
        name='iforest', result_dir='azure_result'):
    # , scale_pos_weight=sum(train_data.fraud_label==1) / sum(train_data.fraud_label==0))
    from sklearn.ensemble import IsolationForest
    from sklearn.metrics import auc, precision_recall_curve, average_precision_score
    ff = IsolationForest(n_estimators=100)
    ff.fit(train_x)  # , labels[train_idx])
    baseline_pred = -ff.score_samples(test_x)

    baseline_pred = np.hstack(
        ((1 - baseline_pred).reshape(-1, 1),
         baseline_pred.reshape(-1, 1)))

    acc, f1, precision, recall, roc_auc, pr_auc, ap, confusion_matrix, roc_r = get_metrics(
        baseline_pred, test_label[test_idx], out_dir=result_dir, name=name, )
    #acc, f1, precision, recall, roc_auc, pr_auc, ap, confusion_matrix = 0, 0, 0, 0,0, 0, 0, 0
    return acc, f1, precision, recall, roc_auc, pr_auc, ap, confusion_matrix, baseline_pred[:, 1], roc_r
