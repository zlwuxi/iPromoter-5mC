#!usr/bin/env python
# -*- coding:utf-8 -*-
# Author:zhanglei Time:2020/3/4 16:12

import numpy as np

"""数据切分函数"""


def split_list_average_n(origin_list, n):
    for i in range(0, len(origin_list), n):
        yield origin_list[i:i + n]

def new_test(num_1):
    X_negative_train = np.load(r'H:\pyworkspace\final_funsion\base('+str(num_1)+r')\onehot\XX_onehot_daluan.npy').reshape(893326,-1)[69750:728611]
    np.save(r'H:\pyworkspace\final_funsion\base('+str(num_1)+r')\onehot\origin\X_negative_1.npy', X_negative_train[:131772])
    np.save(r'H:\pyworkspace\final_funsion\base('+str(num_1)+r')\onehot\origin\X_negative_2.npy', X_negative_train[131772:131772*2])
    np.save(r'H:\pyworkspace\final_funsion\base('+str(num_1)+r')\onehot\origin\X_negative_3.npy', X_negative_train[131772*2:131772*3])
    np.save(r'H:\pyworkspace\final_funsion\base('+str(num_1)+r')\onehot\origin\X_negative_4.npy', X_negative_train[131772*3:131772*4])
    np.save(r'H:\pyworkspace\final_funsion\base('+str(num_1)+r')\onehot\origin\X_negative_5.npy', X_negative_train[131772*4:])
    print('c', X_negative_train[:131772].shape)
    print('c', X_negative_train[131772:131772*2].shape)
    print('c', X_negative_train[131772*2:131772*3].shape)
    print('c', X_negative_train[131772*3:131772*4].shape)
    print('c', X_negative_train[131772*4:].shape)

    y_negative_train = np.load(r'H:\pyworkspace\final_funsion\base('+str(num_1)+r')\onehot\yy_onehot_daluan.npy')[69750:728611]
    np.save(r'H:\pyworkspace\final_funsion\base('+str(num_1)+r')\onehot\origin\y_negative_1.npy', y_negative_train[:131772])
    np.save(r'H:\pyworkspace\final_funsion\base('+str(num_1)+r')\onehot\origin\y_negative_2.npy', y_negative_train[131772:131772*2])
    np.save(r'H:\pyworkspace\final_funsion\base('+str(num_1)+r')\onehot\origin\y_negative_3.npy', y_negative_train[131772*2:131772*3])
    np.save(r'H:\pyworkspace\final_funsion\base('+str(num_1)+r')\onehot\origin\y_negative_4.npy', y_negative_train[131772*3:131772*4])
    np.save(r'H:\pyworkspace\final_funsion\base('+str(num_1)+r')\onehot\origin\y_negative_5.npy', y_negative_train[131772*4:])
    print('d', y_negative_train[:131772].shape)
    print('d', y_negative_train[131772:131772*2].shape)
    print('d', y_negative_train[131772*2:131772*3].shape)
    print('d', y_negative_train[131772*3:131772*4].shape)
    print('d', y_negative_train[131772*4:].shape)

    X_positive_train = np.load(r'H:\pyworkspace\final_funsion\base('+str(num_1)+r')\onehot\XX_onehot_daluan.npy').reshape(893326,-1)[:55800]
    np.save(r'H:\pyworkspace\final_funsion\base('+str(num_1)+r')\onehot\origin\X_positive_1.npy', X_positive_train[:11160])
    np.save(r'H:\pyworkspace\final_funsion\base('+str(num_1)+r')\onehot\origin\X_positive_2.npy', X_positive_train[11160:11160*2])
    np.save(r'H:\pyworkspace\final_funsion\base('+str(num_1)+r')\onehot\origin\X_positive_3.npy', X_positive_train[11160*2:11160*3])
    np.save(r'H:\pyworkspace\final_funsion\base('+str(num_1)+r')\onehot\origin\X_positive_4.npy', X_positive_train[11160*3:11160*4])
    np.save(r'H:\pyworkspace\final_funsion\base('+str(num_1)+r')\onehot\origin\X_positive_5.npy', X_positive_train[11160*4:])
    print('a', X_positive_train[:11160].shape)
    print('a', X_positive_train[11160:11160*2].shape)
    print('a', X_positive_train[11160*2:11160*3].shape)
    print('a', X_positive_train[11160*3:11160*4].shape)
    print('a', X_positive_train[11160*4:].shape)

    y_positive_train = np.load(r'H:\pyworkspace\final_funsion\base('+str(num_1)+r')\onehot\yy_onehot_daluan.npy')[:55800]
    np.save(r'H:\pyworkspace\final_funsion\base('+str(num_1)+r')\onehot\origin\y_positive_1.npy', y_positive_train[:11160])
    np.save(r'H:\pyworkspace\final_funsion\base('+str(num_1)+r')\onehot\origin\y_positive_2.npy', y_positive_train[11160:11160*2])
    np.save(r'H:\pyworkspace\final_funsion\base('+str(num_1)+r')\onehot\origin\y_positive_3.npy', y_positive_train[11160*2:11160*3])
    np.save(r'H:\pyworkspace\final_funsion\base('+str(num_1)+r')\onehot\origin\y_positive_4.npy', y_positive_train[11160*3:11160*4])
    np.save(r'H:\pyworkspace\final_funsion\base('+str(num_1)+r')\onehot\origin\y_positive_5.npy', y_positive_train[11160*4:11160*5])
    print('b', y_positive_train[:11160].shape)
    print('b', y_positive_train[11160:11160*2].shape)
    print('b', y_positive_train[11160*2:11160*3].shape)
    print('b', y_positive_train[11160*3:11160*4].shape)
    print('b', y_positive_train[11160*4:].shape)

# if __name__=="__main__":
#     for i in range(11):
#         new_test(i+10)
