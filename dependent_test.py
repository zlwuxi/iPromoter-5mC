#!usr/bin/env python
# -*- coding:utf-8 -*-
# Author:zhanglei Time:2020/3/12 21:22

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

"""模型生成"""


def model_test(X_test, Y_test, num):
    model = keras.models.load_model(r'H:\pyworkspace\final_funsion\onehot\model\funsion_model' + num + '.h5')
    y_predict = model.predict(X_test)
    y_predict_class = np.argmax(y_predict, axis=1)
    y_test_class = np.argmax(Y_test, axis=1)
    tn, fp, fn, tp = confusion_matrix(y_test_class, y_predict_class).ravel()
    print("TN,FP,FN,TP=", tn, fp, fn, tp)
    fpr, tpr, thresholds = roc_curve(y_test_class, y_predict[:, 1])
    roc_auc = auc(fpr, tpr)  # auc为Roc曲线下的面积
    with open(r'H:\pyworkspace\final_funsion\onehot\onehot_dependent_result.txt',
              'a') as f:
        f.write('base ' + num)
        f.write(
            '\ntn, fp, fn, tp=' + str(tn) + "," + str(fp) + ',' + str(fn) + ',' + str(tp) + ',' + str(roc_auc) + '\n')
    del model


"""总测试"""


def my_main(number):
    X_test = np.load(r'H:\pyworkspace\final_funsion\onehotdependent_X.npy')
    Y_test = np.load(r'H:\pyworkspace\final_funsion\onehotdependent_y.npy')
    model_test(X_test, Y_test, number)

def calc(TN, FP, FN, TP):
    SN = TP / (TP + FN)  # recall
    SP = TN / (TN + FP)
    # Precision = TP / (TP + FP)
    ACC = (TP + TN) / (TP + TN + FN + FP)
    # F1 = (2 * TP) / (2 * TP + FP + FN)
    MCC = (TP * TN - FP * FN) / pow((TP + FN) * (TP + FP) * (TN + FP) * (TN + FN), 0.5)
    return SN, SP, ACC, MCC

TN, FP, FN, TP = [], [], [], []
with open(r'H:\pyworkspace\final_funsion\onehot\onehot_dependent_result.txt', 'r') as f:
    for num, i in enumerate(f.read().splitlines()):
        if num % 2 != 0:
            my=i[15:].split(',')
            TN.append(int(my[0]))
            FP.append(int(my[1]))
            FN.append(int(my[2]))
            TP.append(int(my[3]))

def calc_count():
    with open(r'H:\pyworkspace\final_funsion\onehot\onehot_dependent_result.txt', 'a') as m:
        for k in range(11):
            sn, sp, acc,  mcc = calc(TN[k], FP[k], FN[k], TP[k])
            m.write('sn=%.8f' % sn)
            m.write(',sp=%.8f' % sp)
            m.write(',acc=%.8f' % acc)
            m.write(',mcc=%.8f' % mcc)
            m.write('\n')
#
# calc_count()

if __name__ == "__main__":
    calc_count()
