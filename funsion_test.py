#!usr/bin/env python
# -*- coding:utf-8 -*-
# Author:zhanglei Time:2020/3/4 22:17


import tensorflow as tf

try:
    import tensorflow.python.keras as keras
except:
    import tensorflow.keras as keras

from tensorflow.python.keras import layers
from tensorflow.python.keras import backend as K
from sklearn.metrics import roc_auc_score, auc, roc_curve, matthews_corrcoef
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, f1_score, recall_score, accuracy_score, precision_score
import os
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import math

from tensorflow.python.keras.callbacks import ReduceLROnPlateau

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 只显示error和warining信息 3 只显示error信息
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 这一行注释掉就是使用cpu，不注释就是使用gpu


def calc(TN, FP, FN, TP):
    SN = TP / (TP + FN)  # recall
    SP = TN / (TN + FP)
    # Precision = TP / (TP + FP)
    ACC = (TP + TN) / (TP + TN + FN + FP)
    # F1 = (2 * TP) / (2 * TP + FP + FN)
    fz = TP * TN - FP * FN
    fm = (TP + FN) * (TP + FP) * (TN + FP) * (TN + FN)
    MCC = fz / pow(fm, 0.5)
    return SN, SP, ACC, MCC


def my_main(count, number):
    X_valid = np.load(
        r'H:\pyworkspace\final_funsion\base(' + count + r')\onehot\test_onehot\fold_' + number + '\X_test_fold_' + number + '.npy')
    y_valid = np.load(
        r'H:\pyworkspace\final_funsion\base(' + count + r')\onehot\test_onehot\fold_' + number + '\y_test_fold_' + number + '.npy')
    roc_value = []
    k = 0
    y_predict_class = np.zeros(len(y_valid), dtype=np.float64)
    y_predict_my = np.zeros((len(y_valid), 2), dtype=np.float64)
    for i in range(11):
        model_name = r'H:\pyworkspace\final_funsion\base(' + count + r')\onehot\model\fold_' + number + r'\onehot' + number + '_funsion_' + str(
            k + 1) + '.h5'
        model = keras.models.load_model(model_name)
        y_predict = model.predict(X_valid)
        y_predict_my += y_predict
        y_predict_class += np.argmax(y_predict, axis=1)
        k += 1
        del model
    list = np.zeros(len(y_valid), dtype=np.float64)
    for i, num in enumerate(y_predict_class):
        if num >= 11:
            list[i] = 1
        elif num < 11:
            list[i] = 0
    y_test_class = np.argmax(y_valid, axis=1)
    tn, fp, fn, tp = confusion_matrix(y_test_class, list).ravel()
    print("tn, fp, fn, tp=", tn, fp, fn, tp)
    # sn, sp, acc, mcc = calc(tn, fp, fn, tp)
    fpr, tpr, thresholds = roc_curve(y_test_class, y_predict_my[:, 1])
    roc_auc = auc(fpr, tpr)  # auc为Roc曲线下的面积
    roc_value.append(roc_auc)
    print("AUC=%.6f" % roc_auc)
    with open(r'H:\pyworkspace\final_funsion\onehot\onehot_result.txt',
              'a') as f:
        f.write('base ' + count + ' fold ' + number)
        f.write('\ntn, fp, fn, tp=' + str(tn) + "," + str(fp) + ',' + str(fn) + ',' + str(tp) + '\n')
        # f.write('sn=' + str(sn) + " sp=" + str(sp) + ' acc=' + str(acc) + ' mcc=' + str(mcc) + ' auc=' + str(
        #     roc_auc) + '\n')
    # 开始画ROC曲线
    # plt.plot(fpr, tpr, 'b', label='AUC = %0.3f' % roc_auc)
    # plt.legend(loc='lower right')
    # plt.plot([0, 1], [0, 1], 'r--')
    # plt.xlim([-0.1, 1.1])
    # plt.ylim([-0.1, 1.1])
    # plt.xlabel('False Positive Rate')  # 横坐标是fpr
    # plt.ylabel('True Positive Rate')  # 纵坐标是tpr
    # # plt.title('Receiver operating characteristic example')
    # plt.show()


if __name__ == "__main__":
    for m in range(1):
        for i in range(5):
            my_main(str(m + 20), str(i + 1))
# for i in range(5):
#     my_main(str(20), str(i + 1))
