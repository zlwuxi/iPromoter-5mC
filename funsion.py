#!usr/bin/env python
# -*- coding:utf-8 -*-
# Author:zhanglei Time:2020/3/4 21:06


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
    Precision = TP / (TP + FP)
    ACC = (TP + TN) / (TP + TN + FN + FP)
    F1 = (2 * TP) / (2 * TP + FP + FN)
    MCC = (TP * TN - FP * FN) / pow((TP + FN) * (TP + FP) * (TN + FP) * (TN + FN), 0.5)
    return SN, SP, Precision, ACC, F1, MCC


"""模型生成"""


def model_test(X, Y, dropout, count, num, number):
    model = keras.Sequential()
    model.add(layers.Flatten(input_shape=((int(number)*2+1)*8,)))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(2, activation="sigmoid"))
    model.compile(optimizer="Adam",
                  loss="binary_crossentropy",
                  metrics=['accuracy'])
    model.fit(X, Y, epochs=5, batch_size=256, verbose=0)
    model.save(
        r'H:\pyworkspace\final_funsion\base(' + number + r')\onehot\model\fold_' + count + r'\onehot' + count + '_funsion_' + num + '.h5')
    # y_predict = model.predict(X_test)
    # y_predict_class = np.argmax(y_predict, axis=1)
    # y_test_class = np.argmax(Y_test, axis=1)
    # tn, fp, fn, tp = confusion_matrix(y_test_class, y_predict_class).ravel()
    # sn, sp, precision, acc, f1, mcc = calc(tn, fp, fn, tp)
    # print("TN,FP,FN,TP=", tn, fp, fn, tp)
    # print('SN=%.4f%%' % (sn * 100), 'SP=%.4f%%' % (sp * 100), 'Precision=%.4f%%' % (precision * 100),
    #       'ACC=%.4f%%' % (acc * 100),
    #       'F1_score=%.4f%%' % (f1 * 100), 'MCC=%.4f%%' % (mcc * 100))
    # fpr, tpr, thresholds = roc_curve(y_test_class, y_predict[:, 1])
    # roc_auc = auc(fpr, tpr)  # auc为Roc曲线下的面积
    # # 开始画ROC曲线
    # plt.plot(fpr, tpr, 'b', label='AUC = %0.4f' % roc_auc)
    # plt.legend(loc='lower right')
    # plt.plot([0, 1], [0, 1], 'r--')
    # plt.xlim([-0.1, 1.1])
    # plt.ylim([-0.1, 1.1])
    # plt.xlabel('False Positive Rate')  # 横坐标是fpr
    # plt.ylabel('True Positive Rate')  # 纵坐标是tpr
    # # plt.title('Receiver operating characteristic example')
    # plt.show()
    # del model


"""总测试"""


def my_main(dropout, num_2, num_1, number_):
    X = np.load(
        r'H:\pyworkspace\final_funsion\base(' + number_ + r')\onehot\train_onehot\fold_' + num_2 + r'\traindata\fusion_X_tfold' + num_2 + '_onehot' + num_1 + '.npy')
    Y = np.load(
        r'H:\pyworkspace\final_funsion\base(' + number_ + r')\onehot\train_onehot\fold_' + num_2 + r'\traindata\fusion_y_tfold' + num_2 + '_onehot' + num_1 + '.npy')
    model_test(X, Y, dropout, num_2, num_1, number_)


# if __name__ == "__main__":
#     for m in range(11):
#         for j in range(5):
#             for i in range(11):
#                 my_main(0.3, str(j + 1), str(i + 1), str(m + 10))
