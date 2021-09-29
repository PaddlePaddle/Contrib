# 计算类别权重，先解压数据，然后计算-----已经计算并放入配置文件，无序运行
import numpy as np
import cv2

classes = 20 # 将ignore_index映射成最后一类
normVal = 1.10


def compute_class_weights(histogram):
    classWeights = np.zeros((classes, ))
    normHist = histogram / np.sum(histogram)
    for i in range(classes):
        classWeights[i] = 1 / (np.log(normVal + normHist[i]))
    return classWeights

def readFile(data_dir, fileName):

    global_hist = np.zeros(classes, dtype=np.float32)
    with open(data_dir + '/' + fileName, 'r') as textFile:
        for line in textFile:
            # we expect the text file to contain the data in following format
            # <RGB Image>, <Label Image>
            line_arr = line.split(' ')
            img_file = ((data_dir).strip() + '/' + line_arr[0].strip()).strip()
            label_file = ((data_dir).strip() + '/' + line_arr[1].strip()).strip()
            label_img = cv2.imread(label_file, 0)
            label_img[label_img == 255] = 19

            hist, bins = np.histogram(label_img, classes)
            global_hist += hist

    return compute_class_weights(global_hist)

data_dir = '/home/aistudio/data/data64550/cityscapes'
fileName = 'train.list'
classweight = readFile(data_dir, fileName)
print(classweight[0:-1])