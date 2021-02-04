from __future__ import division
import sys
import random
from PIL import Image
import numpy as np
import numbers
import collections
import traceback

if sys.version_info < (3, 3):
    Sequence = collections.Sequence
    Iterable = collections.Iterable
else:
    Sequence = collections.abc.Sequence
    Iterable = collections.abc.Iterable

def crop(img, top, left, height, width):
    """Crop the given PIL Image.

    Args:
        img (PIL Image): Image to be cropped. (0,0) denotes the top left corner of the image.
        top (int): Vertical component of the top left corner of the crop box.
        left (int): Horizontal component of the top left corner of the crop box.
        height (int): Height of the crop box.
        width (int): Width of the crop box.

    Returns:
        PIL Image: Cropped image.
    """

    return img.crop((left, top, left + width, top + height))



def resize(img, size, interpolation=Image.BILINEAR):
    r"""Resize the input PIL Image to the given size.

    Args:
        img (PIL Image): Image to be resized.
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the smaller edge of the image will be matched to this number maintaining
            the aspect ratio. i.e, if height > width, then image will be rescaled to
            :math:`\left(\text{size} \times \frac{\text{height}}{\text{width}}, \text{size}\right)`
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``

    Returns:
        PIL Image: Resized image.
    """
    if isinstance(size, int):
        w, h = img.size
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return img.resize((ow, oh), interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return img.resize((ow, oh), interpolation)
    else:
        return img.resize(size[::-1], interpolation)


class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *data):
        for f in self.transforms:
            try:
                # multi-fileds in a sample
                if isinstance(data, Sequence):
                    data = f(*data)
                # single field in a sample, call transform directly
                else:
                    data = f(data)
            except Exception as e:
                stack_info = traceback.format_exc()
                print("fail to perform transform [{}] with error: "
                      "{} and stack:\n{}".format(f, e, str(stack_info)))
                raise e
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string



class Resize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image.
        """
        return resize(img, self.size, self.interpolation)



class RandomHorizontalFlip(object):

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img):
        if np.random.random() < self.prob:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img


class Normalize(object):

    def __init__(self, mean=0.5, std=0.5):
        if isinstance(mean, numbers.Number):
            mean = [mean, mean, mean]

        if isinstance(std, numbers.Number):
            std = [std, std, std]

        self.mean = np.array(mean, dtype=np.float32).reshape(len(mean), 1, 1)
        self.std = np.array(std, dtype=np.float32).reshape(len(std), 1, 1)

    def __call__(self, img):
        return (img - self.mean) / self.std


class RandomCrop(object):

    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        i, j, h, w = self.get_params(img, self.size)

        return crop(img, i, j, h, w)


class ToTensor():
    def __init__(self):
        pass

    def __call__(self, img):
        tensor = np.array(img).transpose([2, 0, 1]).astype('float32')
        tensor = tensor / 255.0
        return tensor
