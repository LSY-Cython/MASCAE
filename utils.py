from sklearn import metrics
import numpy as np
import json
from scipy.stats import norm

def classification(train_loss,normal_loss,anomaly_loss,percentile):
    mean, std = norm.fit(train_loss)
    normal_prob = norm.cdf(normal_loss, loc=mean, scale=std)
    anomaly_prob = norm.cdf(anomaly_loss, loc=mean, scale=std)
    rec_thresh = percentile*0.01
    TN = len(np.where(normal_prob < rec_thresh)[0])
    FP = len(normal_prob) - TN
    TP = len(np.where(np.array(anomaly_prob) > rec_thresh)[0])
    FN = len(anomaly_prob) - TP
    return TP, TN, FP, FN, rec_thresh

def evaluation(rec_file):
    with open(rec_file, "r") as f:
        result_content = json.load(f)
    train_loss = result_content["train"]
    normal_loss = result_content["normal"]
    anomaly_loss = result_content["anomaly"]
    fpr_set = list()
    tpr_set = list()  # =recall
    acc_set = list()
    pre_set = list()
    f1_set = list()
    thresh_set = list()
    for per in range(0, 101, 1):
        TP, TN, FP, FN, rec_thresh = classification(train_loss, normal_loss, anomaly_loss, per)
        if TN + FP == 0:
            fpr = 0
        else:
            fpr = FP / (TN + FP)
        if TP + FN == 0:
            tpr = 0
        else:
            tpr = TP / (TP + FN)
        if TP + FP == 0:
            precision = 0
        else:
            precision = TP / (TP + FP)
        if TP + TN + FP + FN == 0:
            accuracy = 0
        else:
            accuracy = (TP + TN) / (TP + TN + FP + FN)
        if precision + tpr != 0:
            f1 = 2 * precision * tpr / (precision + tpr)
        else:
            f1 = 0
        fpr_set.append(fpr)
        tpr_set.append(tpr)
        pre_set.append(precision)
        f1_set.append(f1)
        acc_set.append(accuracy)
        thresh_set.append(rec_thresh)
    opt_index = np.argmax(f1_set)
    accuracy, precision, recall, F1 = acc_set[opt_index], pre_set[opt_index], tpr_set[opt_index], f1_set[opt_index]
    auc = metrics.auc(np.flipud(fpr_set), np.flipud(tpr_set))
    rec_thresh = thresh_set[opt_index]
    print(f"Acc={accuracy}, P={precision}, R={recall}, F1={F1}, AUC={auc}, thresh={rec_thresh}")
