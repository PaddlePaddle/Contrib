from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from PIL import Image
import numpy as np
import transforms
DATASET = "datasets"
A_LIST_FILE = "./data/" + DATASET + "/trainA.txt"
B_LIST_FILE = "./data/" + DATASET + "/trainB.txt"
A_TEST_LIST_FILE = "./data/" + DATASET + "/testA.txt"
B_TEST_LIST_FILE = "./data/" + DATASET + "/testB.txt"
IMAGES_ROOT = "./data/" + DATASET + "/"


def image_shape():
    return [3, 256, 256]


def max_images_num():
    return 3400


def reader_creater(list_file, cycle=True, shuffle=True, return_name=False,transform=None):
    images = [IMAGES_ROOT + line for line in open(list_file, 'r').readlines()]

    def reader():
        while True:
            if shuffle:
                np.random.shuffle(images)
            for file in images:
                file = file.strip("\n\r\t ")
                image = Image.open(file)
                image = image.convert("RGB")
                image = transform(image)
                if return_name:
                    yield image[np.newaxis, :], os.path.basename(file)
                else:
                    yield image
            if not cycle:
                break

    return reader


def a_reader(shuffle=True,transforms=None):
    """
    Reader of images with A style for training.
    """
    return reader_creater(A_LIST_FILE, shuffle=shuffle,transform=transforms)


def b_reader(shuffle=True,transforms=None):
    """
    Reader of images with B style for training.
    """
    return reader_creater(B_LIST_FILE, shuffle=shuffle,transform=transforms)


def a_test_reader(transforms=None):
    """
    Reader of images with A style for test.
    """
    return reader_creater(A_TEST_LIST_FILE, cycle=False, return_name=True,transform=transforms)


def b_test_reader(transforms=None):
    """
    Reader of images with B style for test.
    """
    return reader_creater(B_TEST_LIST_FILE, cycle=False, return_name=True,transform=transforms)
