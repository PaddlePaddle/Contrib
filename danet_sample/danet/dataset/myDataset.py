import os,cv2
from PIL import Image
import numpy as np
import paddle

from my_utils.global_config import *

# 定义数据集
class MyCS_Dataset():
    def __init__(self, imgFiles_list, gtFiles_list,batch_size=None,Img_size=None,Buffer_size=500,mode='train'):
        self.imgFiles_list = imgFiles_list
        self.gtFiles_list = gtFiles_list
        self.size = len(imgFiles_list)
        self.batch_size = batch_size
        self.batch_num = self.size//self.batch_size
        self.Img_size = Img_size
        self.Buffer_size = Buffer_size
        self.mode = mode


    def __len__(self):
        return self.size

    # 定义reader
    def dataset_reader(self):
        def reader():
            if self.mode=='train':
                state = np.random.get_state()
                np.random.shuffle(self.imgFiles_list)
                np.random.set_state(state)
                np.random.shuffle(self.gtFiles_list)         

                for img_path,gt_path in zip(self.imgFiles_list,self.gtFiles_list):
                    img = Image.open(img_path)
                    gt = Image.open(gt_path)
                    yield img, gt
            else:
                for img_path in self.imgFiles_list:
                    img_file = os.path.basename(img_path)[:-4]
                    img = Image.open(img_path)
                    yield img, img_file
 
        return paddle.batch(paddle.reader.xmap_readers(self.transform, reader, process_num = 2, buffer_size =self.Buffer_size),
                            batch_size=self.batch_size)
    
    # 自定义图像变换
    def transform(self, sample):
        def Normalize(image, means, stds):
            for band in range(len(means)):
                image[:, :, band] = (image[:, :, band] - means[band]) / stds[band]
            image = np.transpose(image, [2, 0, 1])
            return image

        image, gt = sample


        if self.mode=='train': 
            x_rd = int(np.random.randint(low=0, high=2048-Img_size, size=1))
            y_rd = int(np.random.randint(low=0, high=1024-Img_size, size=1))
            gt = gt.crop((x_rd,y_rd,x_rd+Img_size,y_rd+Img_size))
            gt = np.array(gt)
            gt = np.eye(Label_num)[gt].transpose((2,0,1)).astype('float32')
        else:
            x_rd = int(1024-Img_size//2)
            y_rd = int(512-Img_size//2)
        image = image.crop((x_rd,y_rd,x_rd+Img_size,y_rd+Img_size))
        image = Normalize(np.array(image).astype('float32'), Means,Stds)
        return image,gt




        