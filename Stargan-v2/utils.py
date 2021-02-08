from __future__ import print_function
import cv2

from six.moves import range
from PIL import Image, ImageOps

import gzip
import numpy as np
import argparse
import struct
import os
import paddle
import random

import paddle.fluid as fluid
from glob import glob


def RandomCrop(img, crop_w, crop_h):

    a = np.random.rand()
    if a > 0.5:
        w, h = img.size[0], img.size[1]

        augment_height_size = h + (30 if h == 256 else int(h * 0.1))
        augment_width_size = w + (30 if w == 256 else int(w * 0.1))

        img = img.resize((augment_height_size, augment_width_size), Image.BILINEAR)
        w, h = img.size[0], img.size[1]
        i = np.random.randint(0, w - crop_w)
        j = np.random.randint(0, h - crop_h)

        img = img.crop((i, j, i + crop_w, j + crop_h))
    return img


def CentorCrop(img, crop_w, crop_h):

    w, h = img.size[0], img.size[1]

    i = int((w - crop_w) / 2.0)
    j = int((h - crop_h) / 2.0)
    a = np.random.rand()
    if a > 0.5:
        img = img.crop((i, j, i + crop_w, j + crop_h))
    return img


def RandomHorizonFlip(img):
    i = np.random.rand()
    if i > 0.5:
        img = ImageOps.mirror(img)
    return img


def get_preprocess_param(load_size, crop_size):
    x = np.random.randint(0, np.maximum(0, load_size - crop_size))
    y = np.random.randint(0, np.maximum(0, load_size - crop_size))
    flip = np.random.rand() > 0.5
    return {
        "crop_pos": (x, y),
        "flip": flip,
        "load_size": load_size,
        "crop_size": crop_size
    }


class Image_data:

    def __init__(self, img_size, channels, dataset_path, domain_list, augment_flag, batch_size):
        self.img_height = img_size
        self.img_width = img_size
        self.channels = channels
        self.augment_flag = augment_flag

        self.dataset_path = dataset_path
        self.domain_list = domain_list
        self.batch_size = batch_size

        self.images = []
        self.shuffle_images = []
        self.domains = []
        self.records = []

    def len(self):
        if self.drop_last or len(self.images) % self.batch_size == 0:
            return len(self.images) // self.batch_size
        else:
            return len(self.images) // self.batch_size + 1

    def image_processing(self, records, images, shuffle_images, domains):
        def reader():
            img_batch = []
            img2_batch = []
            domain_batch = []

            while True:
                print(len(records))
                np.random.shuffle(records)
                print("start get dataset")

                for i in records:
                    img, img2, domain = i

                    img = Image.open(img)  # .convert('RGB')
                    img = img.resize((self.img_height, self.img_width), Image.BILINEAR)

                    if self.augment_flag:
                        img = RandomHorizonFlip(img)
                        img = RandomCrop(img, self.img_height, self.img_width)

                    img = preprocess_fit_train_image(img)
                    img = img.transpose([2, 0, 1])
                    img_batch.append(img)

                    img2 = Image.open(img2)
                    img2 = img2.resize((self.img_height, self.img_width), Image.BILINEAR)

                    if self.augment_flag:
                        img2 = RandomHorizonFlip(img2)
                        img2 = RandomCrop(img2, self.img_height, self.img_width)

                    img2 = preprocess_fit_train_image(img2)
                    img2 = img2.transpose([2, 0, 1])
                    img2_batch.append(img2)
                    domain_batch.append(domain)

                    if len(img_batch) == len(img2_batch) == len(domain_batch) == self.batch_size:
                        yield img_batch, img2_batch, domain_batch
                        img_batch = []
                        img2_batch = []
                        domain_batch = []
                if len(img_batch) == len(img2_batch) == len(domain_batch) != 0:
                    print('len data len is not batch')
                    continue

        return reader()

    def preprocess(self):
        # self.domain_list = ['tiger', 'cat', 'dog', 'lion']
        for idx, domain in enumerate(self.domain_list):
            image_list = glob(os.path.join(self.dataset_path, domain) + '/*.png') + glob(
                os.path.join(self.dataset_path, domain) + '/*.jpg')
            shuffle_list = random.sample(image_list, len(image_list))

            domain_list = [[idx]] * len(image_list)  # [ [0], [0], ... , [0] ]

            self.images.extend(image_list)
            self.shuffle_images.extend(shuffle_list)
            self.domains.extend(domain_list)

        for i in range(len(self.images)):
            record = [self.images[i], self.shuffle_images[i], self.domains[i]]
            self.records.append(record)
        print('len(records)', len(self.records))
        print(self.records[0])


def adjust_dynamic_range(images, range_in, range_out, out_dtype):
    scale = (range_out[1] - range_out[0]) / (range_in[1] - range_in[0])
    bias = range_out[0] - range_in[0] * scale
    images = np.array(images).astype('float32')
    images = images * scale + bias

    images = np.clip(images, range_out[0], range_out[1])
    images = np.cast(images, dtype=out_dtype)
    return images


def preprocess_fit_train_image(images):
    images = (np.array(images).astype('float32') / 255.0 - 0.5) / 0.5
    images = np.clip(images, -1.0, 1.0)

    return images


def postprocess_images(images):
    images = images.numpy()

    images = ((images + 1) * 127.5)
    images = np.clip(images, 0.0, 255.0)

    return images

def load_val_images(image_path, img_size):
    x = Image.open(image_path).convert('RGB')
    img = x.resize((img_size, img_size), Image.BICUBIC)
    img = preprocess_fit_train_image(img)

    img = img.transpose([2, 0, 1])

    img = fluid.dygraph.to_variable(np.array(img))

    return img


def load_images(image_path, img_size, img_channel):
    x = Image.open(image_path).convert('RGB')
    img = x.resize((img_size, img_size), Image.BICUBIC)
    img = preprocess_fit_train_image(img)

    img = img.transpose([2, 0, 1])

    img = fluid.dygraph.to_variable(np.array(img))

    return img


def augmentation(image, augment_height, augment_width):
    image = RandomHorizonFlip(image)
    image = CentorCrop(image, augment_height, augment_width)

    return image


def load_test_image(image_path, img_width, img_height, img_channel):
    if img_channel == 1:
        img = cv2.imread(image_path, flags=cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(image_path, flags=cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, dsize=(img_width, img_height))

    if img_channel == 1:
        img = np.expand_dims(img, axis=0)
        img = np.expand_dims(img, axis=-1)
    else:
        img = np.expand_dims(img, axis=0)

    img = img / 127.5 - 1

    return img


def inverse_transform(images):
    return ((images + 1.) / 2) * 255.0


def imsave(images, size, path):
    images = merge(images, size)
    images = cv2.cvtColor(images.astype('uint8'), cv2.COLOR_RGB2BGR)

    return cv2.imwrite(path, images)


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[h * j:h * (j + 1), w * i:w * (i + 1), :] = image

    return img


def return_images(images, size):
    x = merge(images, size)

    return x


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def str2bool(x):
    return x.lower() in ('true')


def pytorch_xavier_weight_factor(gain=0.02, uniform=False):
    factor = gain * gain
    mode = 'fan_avg'

    return factor, mode, uniform


def pytorch_kaiming_weight_factor(a=0.0, activation_function='relu'):
    if activation_function == 'relu':
        gain = np.sqrt(2.0)
    elif activation_function == 'leaky_relu':
        gain = np.sqrt(2.0 / (1 + a ** 2))
    elif activation_function == 'tanh':
        gain = 5.0 / 3
    else:
        gain = 1.0

    factor = gain * gain
    mode = 'fan_in'

    return factor, mode


def automatic_gpu_usage():
    # 在使用GPU机器时，可以将use_gpu变量设置成True
    use_gpu = True
    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    return place


def soft_update(target, source, decay):
    """
    Copies the parameters from source network (x) to target network (y)
    using the below update
    y = decay * source + (1 - decay) * target_param
    :param target: Target network (PaddleDynaGraphModel)
    :param source: Source network (PaddleDynaGraphModel)
    :decay: decay ratio should be super lower than 1, in range of [0,1]
    :return:
    https://paddlepaddle.org.cn/documentation/docs/zh/api_cn/fluid_cn/Variable_cn.html#set-value
    """
    target_model_map = dict(target.named_parameters())
    for param_name, source_param in source.named_parameters():
        target_param = target_model_map[param_name]
        target_param.set_value(decay * source_param +
                               (1.0 - decay) * target_param)



