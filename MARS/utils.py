#coding=utf-8
from __future__ import division
import csv
import numpy as np
import pdb
import os
import paddle
import paddle.fluid as fluid
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):

    def __init__(self, path, header, resume_path, begin_epoch):
        if (not os.path.exists(path)) or (resume_path==''):
            self.log_file = open(path, 'w+')
            self.logger = csv.writer(self.log_file, delimiter='\t')
            self.logger.writerow(header)
        else:
            self.log_file = open(path, 'r+')
            self.logger = csv.writer(self.log_file, delimiter='\t')
            reader = csv.reader(self.log_file, delimiter='\t')
            lines = []
            print("begin = ", begin_epoch)
            for line in reader:
                lines.append(line)
                if len(lines) == begin_epoch +1 :
                    break
            self.log_file.close()
            self.log_file = open(path, 'w')
            self.logger = csv.writer(self.log_file, delimiter='\t')
            self.logger.writerows(lines[:begin_epoch+1])
            self.log_file.flush()
            
        self.header = header


    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()

class Logger_MARS(object):

    def __init__(self, path, header, resume_path, begin_epoch):
        if resume_path == '':
            self.log_file = open(path, 'w+')
            self.logger = csv.writer(self.log_file, delimiter='\t')
            self.logger.writerow(header)
        else:
            if os.path.exists(path):
                self.log_file = open(path, 'r+')
                self.logger = csv.writer(self.log_file, delimiter='\t')
                reader = csv.reader(self.log_file, delimiter='\t')
                lines = []
                print("begin = ", begin_epoch)
                for line in reader:
                    lines.append(line)
                    if len(lines) == begin_epoch +1 :
                        break
                self.log_file.close()
                self.log_file = open(path, 'w')
                self.logger = csv.writer(self.log_file, delimiter='\t')
                self.logger.writerows(lines[:begin_epoch+1])
                self.log_file.flush()
            else:
                self.log_file = open(path, 'w+')
                self.logger = csv.writer(self.log_file, delimiter='\t')
                self.logger.writerow(header)

        self.header = header


    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value

def calculate_accuracy(outputs, targets):
    #pdb.set_trace()
    batch_size = targets.shape[0]*1.0
    _, pred = fluid.layers.topk(outputs,1)
    pred = fluid.layers.reshape(pred,[-1])
    correct = fluid.layers.equal(pred,targets)#fluid.layers.reshape(targets,[1, -1]))
    correct = fluid.layers.cast(correct, dtype='int64')
    n_correct_elems = fluid.layers.reduce_sum(correct).numpy()
    return n_correct_elems/batch_size
    

