import os
import numpy as np
import pickle

with open('testlist01.txt', 'r') as f:
    paths = f.readlines()
    test_list = [x[:-5] for x in paths]

with open('trainlist01.txt', 'r') as f:
    paths = f.readlines()
    train_list = [x.split(' ')[0][:-4] for x in paths]

label_dir = {}
with open('classInd.txt', 'r') as f:
    label_paths = f.readlines()

###to make the label starting from 0，subtract 1 from each label in classInd.txt
for path in label_paths:
    label_dir[path.split(' ')[1][:-1]] = int(int(path.split(' ')[0]) - 1)

np.save('label_dir.npy', label_dir)

label_dic = np.load('label_dir.npy', allow_pickle=True).item()
print(label_dic)

source_dir = 'data/UCF-101'
target_train_dir = 'data/UCF-101/train'
target_test_dir = 'data/UCF-101/test'

if not os.path.exists(target_train_dir):
    os.mkdir(target_train_dir)
if not os.path.exists(target_test_dir):
    os.mkdir(target_test_dir)


for key in label_dic:
    each_mulu = key + '_jpg'
    print(each_mulu, key) # e.g. IceDancing_jpg IceDancing

    label_dir = os.path.join(source_dir, each_mulu)
    label_mulu = os.listdir(label_dir)

    for each_label_mulu in label_mulu:
        image_file = os.listdir(os.path.join(label_dir, each_label_mulu))
        image_file.sort()
        image_name = image_file[0][:-6]
        image_num = len(image_file)
        frame = []
        vid = image_name
        for i in range(image_num):
            image_path = os.path.join(os.path.join(label_dir, each_label_mulu), image_name + '_' + str(i+1) + '.jpg')
            frame.append(image_path)
        
        path = key +'/' +vid

        output_pkl = vid + '.pkl'
        if path in test_list:
            output_pkl = os.path.join(target_test_dir, output_pkl)
        elif path in train_list:
            output_pkl = os.path.join(target_train_dir, output_pkl)

        f = open(output_pkl, 'wb')
        pickle.dump((vid, label_dic[key], frame), f, -1)
        f.close()
