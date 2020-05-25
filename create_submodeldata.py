#!usr/bin/env python
# -*- coding:utf-8 -*-
# Author:zhanglei Time:2020/3/4 18:25

"""
Required files before submodel generation.
"""

import numpy as np
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
from onehot.new_test import new_test
import math

from tensorflow.python.keras.callbacks import ReduceLROnPlateau

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Only error and warining information are displayed
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # To comment out this line is to use the CPU, not to comment out is to use the gpu


def calc(TN, FP, FN, TP):
    SN = TP / (TP + FN)  # recall
    SP = TN / (TN + FP)
    Precision = TP / (TP + FP)
    ACC = (TP + TN) / (TP + TN + FN + FP)
    F1 = (2 * TP) / (2 * TP + FP + FN)
    MCC = (TP * TN - FP * FN) / pow((TP + FN) * (TP + FP) * (TN + FP) * (TN + FN), 0.5)
    return SN, SP, Precision, ACC, F1, MCC


def split_list_average_n(origin_list, n):
    for i in range(0, len(origin_list), n):
        yield origin_list[i:i + n]

def made_1x(count):
    positive = np.vstack([np.load(
        r'H:\pyworkspace\final_funsion\base(' + str(count) + r')\onehot\origin\X_positive_2.npy'),
        np.load(r'H:\pyworkspace\final_funsion\base(' + str(
            count) + r')\onehot\origin\X_positive_3.npy'),
        np.load(r'H:\pyworkspace\final_funsion\base(' + str(
            count) + r')\onehot\origin\X_positive_4.npy'),
        np.load(r'H:\pyworkspace\final_funsion\base(' + str(
            count) + r')\onehot\origin\X_positive_5.npy')])
    negative = np.vstack([np.load(
        r'H:\pyworkspace\final_funsion\base(' + str(count) + r')\onehot\origin\X_negative_2.npy'),
        np.load(r'H:\pyworkspace\final_funsion\base(' + str(
            count) + r')\onehot\origin\X_negative_3.npy'),
        np.load(r'H:\pyworkspace\final_funsion\base(' + str(
            count) + r')\onehot\origin\X_negative_4.npy'),
        np.load(r'H:\pyworkspace\final_funsion\base(' + str(
            count) + r')\onehot\origin\X_negative_5.npy')])

    np.save(r'H:\pyworkspace\final_funsion\base(' + str(
        count) + r')\onehot\train_onehot\fold_1\X_train_positive_fold_1.npy', positive)
    np.save(r'H:\pyworkspace\final_funsion\base(' + str(
        count) + r')\onehot\train_onehot\fold_1\X_train_negative_fold_1.npy', negative)


def made_2x(count):
    positive = np.vstack([np.load(
        r'H:\pyworkspace\final_funsion\base(' + str(count) + r')\onehot\origin\X_positive_1.npy'),
        np.load(r'H:\pyworkspace\final_funsion\base(' + str(
            count) + r')\onehot\origin\X_positive_3.npy'),
        np.load(r'H:\pyworkspace\final_funsion\base(' + str(
            count) + r')\onehot\origin\X_positive_4.npy'),
        np.load(r'H:\pyworkspace\final_funsion\base(' + str(
            count) + r')\onehot\origin\X_positive_5.npy')])
    negative = np.vstack([np.load(
        r'H:\pyworkspace\final_funsion\base(' + str(count) + r')\onehot\origin\X_negative_1.npy'),
        np.load(r'H:\pyworkspace\final_funsion\base(' + str(
            count) + r')\onehot\origin\X_negative_3.npy'),
        np.load(r'H:\pyworkspace\final_funsion\base(' + str(
            count) + r')\onehot\origin\X_negative_4.npy'),
        np.load(r'H:\pyworkspace\final_funsion\base(' + str(
            count) + r')\onehot\origin\X_negative_5.npy')])

    np.save(r'H:\pyworkspace\final_funsion\base(' + str(
        count) + r')\onehot\train_onehot\fold_2\X_train_positive_fold_2.npy', positive)
    np.save(r'H:\pyworkspace\final_funsion\base(' + str(
        count) + r')\onehot\train_onehot\fold_2\X_train_negative_fold_2.npy', negative)


def made_3x(count):
    positive = np.vstack([np.load(
        r'H:\pyworkspace\final_funsion\base(' + str(count) + r')\onehot\origin\X_positive_1.npy'),
        np.load(r'H:\pyworkspace\final_funsion\base(' + str(
            count) + r')\onehot\origin\X_positive_2.npy'),
        np.load(r'H:\pyworkspace\final_funsion\base(' + str(
            count) + r')\onehot\origin\X_positive_4.npy'),
        np.load(r'H:\pyworkspace\final_funsion\base(' + str(
            count) + r')\onehot\origin\X_positive_5.npy')])
    negative = np.vstack([np.load(
        r'H:\pyworkspace\final_funsion\base(' + str(count) + r')\onehot\origin\X_negative_1.npy'),
        np.load(r'H:\pyworkspace\final_funsion\base(' + str(
            count) + r')\onehot\origin\X_negative_2.npy'),
        np.load(r'H:\pyworkspace\final_funsion\base(' + str(
            count) + r')\onehot\origin\X_negative_4.npy'),
        np.load(r'H:\pyworkspace\final_funsion\base(' + str(
            count) + r')\onehot\origin\X_negative_5.npy')])

    np.save(r'H:\pyworkspace\final_funsion\base(' + str(
        count) + r')\onehot\train_onehot\fold_3\X_train_positive_fold_3.npy', positive)
    np.save(r'H:\pyworkspace\final_funsion\base(' + str(
        count) + r')\onehot\train_onehot\fold_3\X_train_negative_fold_3.npy', negative)


def made_4x(count):
    positive = np.vstack([np.load(
        r'H:\pyworkspace\final_funsion\base(' + str(count) + r')\onehot\origin\X_positive_1.npy'),
        np.load(r'H:\pyworkspace\final_funsion\base(' + str(
            count) + r')\onehot\origin\X_positive_2.npy'),
        np.load(r'H:\pyworkspace\final_funsion\base(' + str(
            count) + r')\onehot\origin\X_positive_3.npy'),
        np.load(r'H:\pyworkspace\final_funsion\base(' + str(
            count) + r')\onehot\origin\X_positive_5.npy')])
    negative = np.vstack([np.load(
        r'H:\pyworkspace\final_funsion\base(' + str(count) + r')\onehot\origin\X_negative_1.npy'),
        np.load(r'H:\pyworkspace\final_funsion\base(' + str(
            count) + r')\onehot\origin\X_negative_2.npy'),
        np.load(r'H:\pyworkspace\final_funsion\base(' + str(
            count) + r')\onehot\origin\X_negative_3.npy'),
        np.load(r'H:\pyworkspace\final_funsion\base(' + str(
            count) + r')\onehot\origin\X_negative_5.npy')])

    np.save(r'H:\pyworkspace\final_funsion\base(' + str(
        count) + r')\onehot\train_onehot\fold_4\X_train_positive_fold_4.npy', positive)
    np.save(r'H:\pyworkspace\final_funsion\base(' + str(
        count) + r')\onehot\train_onehot\fold_4\X_train_negative_fold_4.npy', negative)


def made_5x(count):
    positive = np.vstack([np.load(
        r'H:\pyworkspace\final_funsion\base(' + str(count) + r')\onehot\origin\X_positive_1.npy'),
        np.load(r'H:\pyworkspace\final_funsion\base(' + str(
            count) + r')\onehot\origin\X_positive_2.npy'),
        np.load(r'H:\pyworkspace\final_funsion\base(' + str(
            count) + r')\onehot\origin\X_positive_3.npy'),
        np.load(r'H:\pyworkspace\final_funsion\base(' + str(
            count) + r')\onehot\origin\X_positive_4.npy')])
    negative = np.vstack([np.load(
        r'H:\pyworkspace\final_funsion\base(' + str(count) + r')\onehot\origin\X_negative_1.npy'),
        np.load(r'H:\pyworkspace\final_funsion\base(' + str(
            count) + r')\onehot\origin\X_negative_2.npy'),
        np.load(r'H:\pyworkspace\final_funsion\base(' + str(
            count) + r')\onehot\origin\X_negative_3.npy'),
        np.load(r'H:\pyworkspace\final_funsion\base(' + str(
            count) + r')\onehot\origin\X_negative_4.npy')])

    np.save(r'H:\pyworkspace\final_funsion\base(' + str(
        count) + r')\onehot\train_onehot\fold_5\X_train_positive_fold_5.npy', positive)
    np.save(r'H:\pyworkspace\final_funsion\base(' + str(
        count) + r')\onehot\train_onehot\fold_5\X_train_negative_fold_5.npy', negative)


def made_1y(count):
    positive = np.vstack([np.load(
        r'H:\pyworkspace\final_funsion\base(' + str(count) + r')\onehot\origin\y_positive_2.npy'),
        np.load(r'H:\pyworkspace\final_funsion\base(' + str(
            count) + r')\onehot\origin\y_positive_3.npy'),
        np.load(r'H:\pyworkspace\final_funsion\base(' + str(
            count) + r')\onehot\origin\y_positive_4.npy'),
        np.load(r'H:\pyworkspace\final_funsion\base(' + str(
            count) + r')\onehot\origin\y_positive_5.npy')])
    negative = np.vstack([np.load(
        r'H:\pyworkspace\final_funsion\base(' + str(count) + r')\onehot\origin\y_negative_2.npy'),
        np.load(r'H:\pyworkspace\final_funsion\base(' + str(
            count) + r')\onehot\origin\y_negative_3.npy'),
        np.load(r'H:\pyworkspace\final_funsion\base(' + str(
            count) + r')\onehot\origin\y_negative_4.npy'),
        np.load(r'H:\pyworkspace\final_funsion\base(' + str(
            count) + r')\onehot\origin\y_negative_5.npy')])

    np.save(r'H:\pyworkspace\final_funsion\base(' + str(
        count) + r')\onehot\train_onehot\fold_1\y_train_positive_fold_1.npy', positive)
    np.save(r'H:\pyworkspace\final_funsion\base(' + str(
        count) + r')\onehot\train_onehot\fold_1\y_train_negative_fold_1.npy', negative)


def made_2y(count):
    positive = np.vstack([np.load(
        r'H:\pyworkspace\final_funsion\base(' + str(count) + r')\onehot\origin\y_positive_1.npy'),
        np.load(r'H:\pyworkspace\final_funsion\base(' + str(
            count) + r')\onehot\origin\y_positive_3.npy'),
        np.load(r'H:\pyworkspace\final_funsion\base(' + str(
            count) + r')\onehot\origin\y_positive_4.npy'),
        np.load(r'H:\pyworkspace\final_funsion\base(' + str(
            count) + r')\onehot\origin\y_positive_5.npy')])
    negative = np.vstack([np.load(
        r'H:\pyworkspace\final_funsion\base(' + str(count) + r')\onehot\origin\y_negative_1.npy'),
        np.load(r'H:\pyworkspace\final_funsion\base(' + str(
            count) + r')\onehot\origin\y_negative_3.npy'),
        np.load(r'H:\pyworkspace\final_funsion\base(' + str(
            count) + r')\onehot\origin\y_negative_4.npy'),
        np.load(r'H:\pyworkspace\final_funsion\base(' + str(
            count) + r')\onehot\origin\y_negative_5.npy')])

    np.save(r'H:\pyworkspace\final_funsion\base(' + str(
        count) + r')\onehot\train_onehot\fold_2\y_train_positive_fold_2.npy', positive)
    np.save(r'H:\pyworkspace\final_funsion\base(' + str(
        count) + r')\onehot\train_onehot\fold_2\y_train_negative_fold_2.npy', negative)


def made_3y(count):
    positive = np.vstack([np.load(
        r'H:\pyworkspace\final_funsion\base(' + str(count) + r')\onehot\origin\y_positive_1.npy'),
        np.load(r'H:\pyworkspace\final_funsion\base(' + str(
            count) + r')\onehot\origin\y_positive_2.npy'),
        np.load(r'H:\pyworkspace\final_funsion\base(' + str(
            count) + r')\onehot\origin\y_positive_4.npy'),
        np.load(r'H:\pyworkspace\final_funsion\base(' + str(
            count) + r')\onehot\origin\y_positive_5.npy')])
    negative = np.vstack([np.load(
        r'H:\pyworkspace\final_funsion\base(' + str(count) + r')\onehot\origin\y_negative_1.npy'),
        np.load(r'H:\pyworkspace\final_funsion\base(' + str(
            count) + r')\onehot\origin\y_negative_2.npy'),
        np.load(r'H:\pyworkspace\final_funsion\base(' + str(
            count) + r')\onehot\origin\y_negative_4.npy'),
        np.load(r'H:\pyworkspace\final_funsion\base(' + str(
            count) + r')\onehot\origin\y_negative_5.npy')])

    np.save(r'H:\pyworkspace\final_funsion\base(' + str(
        count) + r')\onehot\train_onehot\fold_3\y_train_positive_fold_3.npy', positive)
    np.save(r'H:\pyworkspace\final_funsion\base(' + str(
        count) + r')\onehot\train_onehot\fold_3\y_train_negative_fold_3.npy', negative)


def made_4y(count):
    positive = np.vstack([np.load(
        r'H:\pyworkspace\final_funsion\base(' + str(count) + r')\onehot\origin\y_positive_1.npy'),
        np.load(r'H:\pyworkspace\final_funsion\base(' + str(
            count) + r')\onehot\origin\y_positive_2.npy'),
        np.load(r'H:\pyworkspace\final_funsion\base(' + str(
            count) + r')\onehot\origin\y_positive_3.npy'),
        np.load(r'H:\pyworkspace\final_funsion\base(' + str(
            count) + r')\onehot\origin\y_positive_5.npy')])
    negative = np.vstack([np.load(
        r'H:\pyworkspace\final_funsion\base(' + str(count) + r')\onehot\origin\y_negative_1.npy'),
        np.load(r'H:\pyworkspace\final_funsion\base(' + str(
            count) + r')\onehot\origin\y_negative_2.npy'),
        np.load(r'H:\pyworkspace\final_funsion\base(' + str(
            count) + r')\onehot\origin\y_negative_3.npy'),
        np.load(r'H:\pyworkspace\final_funsion\base(' + str(
            count) + r')\onehot\origin\y_negative_5.npy')])

    np.save(r'H:\pyworkspace\final_funsion\base(' + str(
        count) + r')\onehot\train_onehot\fold_4\y_train_positive_fold_4.npy', positive)
    np.save(r'H:\pyworkspace\final_funsion\base(' + str(
        count) + r')\onehot\train_onehot\fold_4\y_train_negative_fold_4.npy', negative)


def made_5y(count):
    positive = np.vstack([np.load(
        r'H:\pyworkspace\final_funsion\base(' + str(count) + r')\onehot\origin\y_positive_1.npy'),
        np.load(r'H:\pyworkspace\final_funsion\base(' + str(
            count) + r')\onehot\origin\y_positive_2.npy'),
        np.load(r'H:\pyworkspace\final_funsion\base(' + str(
            count) + r')\onehot\origin\y_positive_3.npy'),
        np.load(r'H:\pyworkspace\final_funsion\base(' + str(
            count) + r')\onehot\origin\y_positive_4.npy')])
    negative = np.vstack([np.load(
        r'H:\pyworkspace\final_funsion\base(' + str(count) + r')\onehot\origin\y_negative_1.npy'),
        np.load(r'H:\pyworkspace\final_funsion\base(' + str(
            count) + r')\onehot\origin\y_negative_2.npy'),
        np.load(r'H:\pyworkspace\final_funsion\base(' + str(
            count) + r')\onehot\origin\y_negative_3.npy'),
        np.load(r'H:\pyworkspace\final_funsion\base(' + str(
            count) + r')\onehot\origin\y_negative_4.npy')])

    np.save(r'H:\pyworkspace\final_funsion\base(' + str(
        count) + r')\onehot\train_onehot\fold_5\y_train_positive_fold_5.npy', positive)
    np.save(r'H:\pyworkspace\final_funsion\base(' + str(
        count) + r')\onehot\train_onehot\fold_5\y_train_negative_fold_5.npy', negative)


def create(num_1, num):
    positive = np.load(r'H:\pyworkspace\final_funsion\base(' + str(
        num_1) + r')\onehot\origin\X_positive_' + num + '.npy')
    negative = np.load(r'H:\pyworkspace\final_funsion\base(' + str(
        num_1) + r')\onehot\origin\X_negative_' + num + '.npy')
    positive_y = np.load(r'H:\pyworkspace\final_funsion\base(' + str(
        num_1) + r')\onehot\origin\y_positive_' + num + '.npy')
    negative_y = np.load(r'H:\pyworkspace\final_funsion\base(' + str(
        num_1) + r')\onehot\origin\y_negative_' + num + '.npy')
    a = np.vstack([positive, negative])
    b = np.vstack([positive_y, negative_y])
    state_1 = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(state_1)
    np.random.shuffle(b)

    np.save(r'H:\pyworkspace\final_funsion\base(' + str(
        num_1) + r')\onehot\test_onehot\fold_' + num + '\X_test_fold_' + num + '.npy', a)
    np.save(r'H:\pyworkspace\final_funsion\base(' + str(
        num_1) + r')\onehot\test_onehot\fold_' + num + '\y_test_fold_' + num + '.npy', b)

    if num == str(1):
        made_1x(num_1)
        made_1y(num_1)
    elif num == str(2):
        made_2x(num_1)
        made_2y(num_1)
    elif num == str(3):
        made_3x(num_1)
        made_3y(num_1)
    elif num == str(4):
        made_4x(num_1)
        made_4y(num_1)
    elif num == str(5):
        made_5x(num_1)
        made_5y(num_1)


"""Training data preparation"""


def funsion_model_traindata_prepare(num_2, num):
    X_negative_train = np.load(r'H:\pyworkspace\final_funsion\base(' + str(
        num_2) + r')\onehot\train_onehot\fold_' + num + '\X_train_negative_fold_' + num + '.npy')
    X_positive_train = np.load(r'H:\pyworkspace\final_funsion\base(' + str(
        num_2) + r')\onehot\train_onehot\fold_' + num + '\X_train_positive_fold_' + num + '.npy')
    y_negative_train = np.load(r'H:\pyworkspace\final_funsion\base(' + str(
        num_2) + r')\onehot\train_onehot\fold_' + num + '\y_train_negative_fold_' + num + '.npy')
    y_positive_train = np.load(r'H:\pyworkspace\final_funsion\base(' + str(
        num_2) + r')\onehot\train_onehot\fold_' + num + '\y_train_positive_fold_' + num + '.npy')

    b = split_list_average_n(X_negative_train, 47917)
    c = split_list_average_n(y_negative_train, 47917)  # 68624
    list_1 = []
    list_2 = []
    for seq in b:
        list_1.append(np.vstack([X_positive_train, seq]))
    for x in c:
        list_2.append(np.vstack([y_positive_train, x]))
    k = 0
    for i in range(11):
        name_X = r'H:\pyworkspace\final_funsion\base(' + str(
            num_2) + r')\onehot\train_onehot\fold_' + num + r'\traindata\fusion_X_tfold' + num + '_onehot' + str(
            k + 1)
        name_y = r'H:\pyworkspace\final_funsion\base(' + str(
            num_2) + r')\onehot\train_onehot\fold_' + num + r'\traindata\fusion_y_tfold' + num + '_onehot' + str(
            k + 1)
        XX = list_1[i]
        yy = list_2[i]
        state_1 = np.random.get_state()
        np.random.shuffle(XX)
        np.random.set_state(state_1)
        np.random.shuffle(yy)
        np.save(name_X, XX)
        np.save(name_y, yy)
        k += 1

def model_test(X, Y, dropout, count, num, number):
    model = keras.Sequential()
    model.add(layers.Flatten(input_shape=((int(number)*2+1)*4,)))
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

def my_main(dropout, num_2, num_1, number_):
    X = np.load(
        r'H:\pyworkspace\final_funsion\base(' + number_ + r')\onehot\train_onehot\fold_' + num_2 + r'\traindata\fusion_X_tfold' + num_2 + '_onehot' + num_1 + '.npy')
    Y = np.load(
        r'H:\pyworkspace\final_funsion\base(' + number_ + r')\onehot\train_onehot\fold_' + num_2 + r'\traindata\fusion_y_tfold' + num_2 + '_onehot' + num_1 + '.npy')
    model_test(X, Y, dropout, num_2, num_1, number_)

if __name__ == "__main__":
    for h in range(11):
        new_test(h+10)
    for base in range(11):
        for i in range(5):
            create(str(base+10), str(i + 1))
        for j in range(5):
            funsion_model_traindata_prepare(str(base+10), str(j + 1))
    for m in range(11):
        for k in range(5):
            for p in range(11):
                my_main(0.3, str(k + 1), str(p + 1), str(m + 10))
