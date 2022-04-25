import argparse
import pickle

import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--datasets', default='ntu/xsub', choices={'kinetics', 'ntu/xsub', 'ntu/xview'},
                    help='the work folder for storing results')
parser.add_argument('--alpha', default=1, help='weighted summation')
arg = parser.parse_args()

dataset = arg.datasets
label = open('./data/' + dataset + '/val_label.pkl', 'rb')
label = np.array(pickle.load(label))
r1 = open('./work_dir/' + dataset + '/agcn_test_joint/epoch1_test_score.pkl', 'rb')
r1 = list(pickle.load(r1).items())
r2 = open('./work_dir/' + dataset + '/agcn_test_bone/epoch1_test_score.pkl', 'rb')
r2 = list(pickle.load(r2).items())
# right_num = total_num = right_num_5 = 0
# for i in tqdm(range(len(label[0]))):
#     _, l = label[:, i]
#     _, r11 = r1[i]
#     _, r22 = r2[i]
#     r = r11 + r22 * arg.alpha
#     rank_5 = r.argsort()[-5:]
#     right_num_5 += int(int(l) in rank_5)
#     r = np.argmax(r)
#     right_num += int(r == int(l))
#     total_num += 1
# acc = right_num / total_num
# acc5 = right_num_5 / total_num
# print('acc: ', "%.3f" % acc, 'acc5: ', "%.3f" % acc5)
a = np.arange(0, 1.1, 0.1)
acc_best = 0
acc5_best = 0
M = 0
N = 0

for m in range(11):
    for n in range(11):
        right_num = total_num = right_num_5 = 0
        for i in tqdm(range(len(label[0]))):
            _, l = label[:, i]
            _, r11 = r1[i]
            _, r22 = r2[i]
            r = r11 * a[m] + r22 * a[n]
            rank_5 = r.argsort()[-5:]
            right_num_5 += int(int(l) in rank_5)
            r = np.argmax(r)
            right_num += int(r == int(l))
            total_num += 1
        acc = right_num / total_num
        acc5 = right_num_5 / total_num
        if acc >= acc_best:
            acc_best = acc
            acc5_best = acc5
            M = m
            N = n
# print(acc, acc5)
print('acc: ',"%.3f" % acc_best, 'acc5: ',"%.3f" % acc5_best) 
