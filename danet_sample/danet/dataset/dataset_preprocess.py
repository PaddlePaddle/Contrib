import os, cv2
import numpy as np
import pandas as pd
import sys
sys.path.append('/home/aistudio/work/paddle/danet')
from my_utils.global_config import *


def get_setCSV(datasetImg_path=Images_path,
                datasetGT_path=GTs_path,
                datasetSave_path=Split_csv_path):
    # 生成img_path gt_path列表csv
    datasets = ['train','val','test']
    imgs_list = []
    gts_list = []
    split_list = []
    for dataset in datasets:
        print(dataset,'statistic...')

        dataImg_path = os.path.join(datasetImg_path, dataset)
        dataGT_path = os.path.join(datasetGT_path, dataset)
        citys = os.listdir(dataImg_path)
        for city in citys:
            cityImg_path = os.path.join(dataImg_path,city)
            cityGT_path = os.path.join(dataGT_path,city)
            cityImgs_list = os.listdir(cityImg_path)
            cityImgs_path_list = [os.path.join(cityImg_path,img_name) for img_name in cityImgs_list]
            cityGTs_path_list = [os.path.join(cityGT_path,img_name.replace('leftImg8bit','gtFine_labelTrainIds')) for img_name in cityImgs_list]
            imgs_list += cityImgs_path_list
            gts_list += cityGTs_path_list
            split_list += [dataset]*len(cityImgs_list)
            print(dataset,city)
    split_dict = {'img_path': imgs_list,
            'gt_path': gts_list,
            'split': split_list}
    df = pd.DataFrame(split_dict)
    df.to_csv(datasetSave_path)
    print('save to', datasetSave_path)




if __name__ == '__main__':
    pass
    get_setCSV()




