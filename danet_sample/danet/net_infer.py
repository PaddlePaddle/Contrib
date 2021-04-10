import os,cv2,glob
import numpy as np
import pandas as pd
import seaborn as sns
from paddle import fluid
import matplotlib.pyplot as plt


from dataset.myDataset import *
from networks.unet import *
from networks.unet_sample import *
from my_utils.com_utils import *
from my_utils.global_config import *


def infer_app():
        # 读取数据集
    df = pd.read_csv(Split_csv_path, header=0, index_col=0)

    imgs_list = df[df['split'] == 'test']['img_path'].tolist()

    # imgs_list = glob.glob(os.path.join(Infer_Imgs_path,'*.tif'))
    inference(imgs_list,Infer_path,Model_path)
    

def inference(imgs_list, pres_save_path, model_path):
    '''
    常用函数，对图片进行推理
    :param imgs_list: 图片名列表
    :param imgs_path: 图片所在文件夹的路径
    :param pres_save_path:图片预测结果文件夹的路径
    :param model_path:
    :return:
    '''
    # 数据加载
    print('dataset load...')
    test_dataset = MyCS_Dataset(imgs_list, None,batch_size=Batch_size,Buffer_size=Batch_size*10,mode='val')
    test_reader = test_dataset.dataset_reader()


    print('model load...')
    place = fluid.CUDAPlace(0) if Use_gpu else fluid.CPUPlace()
    with fluid.dygraph.guard(place):
        model = DANet('danet')

        # 读取上次保存的模型
        if os.path.exists(model_path+'.pdparams'):
            print(model_path,'reload...')
            m_para, m_opt = fluid.load_dygraph(model_path)
            model.load_dict(m_para)
        else:
            print('wrong!')
            return

        # 预测输出
        model.eval()
        for batch_num, samples in enumerate(test_reader()):  
            # 数据格式转换
            images = np.array([sample[0] for sample in samples])
            imgs_name = np.array([sample[1] for sample in samples])
            images = fluid.dygraph.to_variable(images)

            # 前向传播
            p_out, c_out,output = model(images)
            # output = model(images)

            pres = np.argmax(output.numpy(), axis=1)

            for index in range(pres.shape[0]):
                file_name = imgs_name[index]
                pre_save_path = os.path.join(pres_save_path, file_name.replace('leftImg8bit','gtFine_labelTrainIds')+'.png')
                cv2.imwrite(pre_save_path, pres[index])
            process_show(batch_num,val_dataset.batch_num,'infer:')
        print()

if __name__ == '__main__':
    pass
    infer_app()




