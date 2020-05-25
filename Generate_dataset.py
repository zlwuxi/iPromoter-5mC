#!usr/bin/env python
# -*- coding:utf-8 -*-
# Author:zhanglei Time:2020/1/4 15:36

from Bio import SeqIO
import numpy as np

def AA_ONE_HOT(AA):
    one_hot_dict = {
        'A': [1, 0, 0, 0],
        'C': [0, 1, 0, 0],
        'G': [0, 0, 1, 0],
        'T': [0, 0, 0, 1],
        'N': [0, 0, 0, 0]
    }

    coding_arr = np.zeros((len(AA), 4), dtype=float)

    for m in range(len(AA)):
        coding_arr[m] = one_hot_dict[AA[m]]

    return coding_arr


i = 0
j = 0
a = np.zeros((69750, 41, 4))
a_label = np.zeros((69750, 2))
b = np.zeros((823576, 41, 4))
b_label = np.zeros((823576, 2))
aa = a.copy()
aa_label = a_label.copy()
bb = b.copy()
bb_label = b_label.copy()
for my_aa in SeqIO.parse(r'D:\pyworkspace\funsion_desion_model\origin_data\all_positive.fasta', 'fasta'):
    AA = str(my_aa.seq)
    aa[i] = AA_ONE_HOT(AA)
    aa_label[i] = [0,1]
    i += 1

for my_bb in SeqIO.parse(r'D:\pyworkspace\funsion_desion_model\origin_data\all_negative.fasta', 'fasta'):
    AA = str(my_bb.seq)
    bb[j] = AA_ONE_HOT(AA)
    bb_label[j] = [1,0]
    j += 1

np.random.shuffle(aa)
np.random.shuffle(bb)
X = np.vstack([aa, bb])
y = np.vstack([aa_label, bb_label])
np.save('XX_onehot_daluan.npy', X)
np.save('yy_onehot_daluan.npy', y)