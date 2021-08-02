#encoding=utf8
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import cv2
import numpy as np
import random

data_folder = './data/omniglot'  # omniglot数据集路径

character_folders = [os.path.join(data_folder, family, character) \
                     for family in os.listdir(data_folder) \
                     if os.path.isdir(os.path.join(data_folder, family)) \
                     for character in os.listdir(os.path.join(data_folder, family))]
print("The number of character folders: {}".format(len(character_folders)))  # 1623
random.seed(1)
random.shuffle(character_folders)
train_folders = character_folders[:973]
val_folders = character_folders[973:1298]
test_folders = character_folders[1298:]
print('The number of train characters is {}'.format(len(train_folders)))  # 973
print('The number of validation characters is {}'.format(len(val_folders)))  # 325
print('The number of test characters is {}'.format(len(test_folders)))  # 325

train_imgs_list = []
for char_fold in train_folders:
    char_list = []
    for file in [os.path.join(char_fold, f) for f in os.listdir(char_fold)]:
        img = cv2.imread(file)
        img = cv2.resize(img, (28, 28))
        img = np.transpose(img, (2, 0, 1))
        img = img[0].astype('float32')  # 只取零通道
        img = img / 255.0
        img = img * 2.0 - 1.0
        char_list.append(img)
    char_list = np.array(char_list)
    train_imgs_list.append(char_list)
train_imgs = np.array(train_imgs_list)
train_imgs = train_imgs[:, :, np.newaxis, :, :]
print('The shape of train_imgs: {}'.format(train_imgs.shape))  # [973,20,1,28,28]

val_imgs_list = []
for char_fold in val_folders:
    char_list = []
    for file in [os.path.join(char_fold, f) for f in os.listdir(char_fold)]:
        img = cv2.imread(file)
        img = cv2.resize(img, (28, 28))
        img = np.transpose(img, (2, 0, 1))
        img = img[0].astype('float32')  # 只取零通道
        img = img / 255.0
        img = img * 2.0 - 1.0
        char_list.append(img)
    char_list = np.array(char_list)
    val_imgs_list.append(char_list)
val_imgs = np.array(val_imgs_list)
val_imgs = val_imgs[:, :, np.newaxis, :, :]
print('The shape of val_imgs: {}'.format(val_imgs.shape))  # [325,20,1,28,28]

test_imgs_list = []
for char_fold in test_folders:
    char_list = []
    for file in [os.path.join(char_fold, f) for f in os.listdir(char_fold)]:
        img = cv2.imread(file)
        img = cv2.resize(img, (28, 28))
        img = np.transpose(img, (2, 0, 1))
        img = img[0].astype('float32')  # 只取零通道
        img = img / 255.0
        img = img * 2.0 - 1.0
        char_list.append(img)
    char_list = np.array(char_list)
    test_imgs_list.append(char_list)
test_imgs = np.array(test_imgs_list)
test_imgs = test_imgs[:, :, np.newaxis, :, :]
print('The shape of test_imgs: {}'.format(test_imgs.shape))  # [325,20,1,28,28]

np.save(os.path.join('omniglot_train.npy'), train_imgs)
np.save(os.path.join('omniglot_val.npy'), val_imgs)
np.save(os.path.join('omniglot_test.npy'), test_imgs)

