import os,cv2
import numpy as np
import pandas as pd
import seaborn as sns
from paddle import fluid
import matplotlib.pyplot as plt


from dataset.myDataset import *
from networks.danet import *
from my_utils.com_utils import *
from my_utils.global_config import *


def test():
    # 读取数据集
    df = pd.read_csv(Split_csv_path, header=0, index_col=0)

    valImg_list = df[df['split'] == 'val']['img_path'].tolist()
    valGT_list = df[df['split'] == 'val']['gt_path'].tolist()
    valPre_list = [os.path.join(Pres_path,os.path.basename(img_path)[:-4]+'.png') for img_path in valImg_list]

    inference(valImg_list, Pres_path, Model_path)

    # 指标计算
    cfm = mean_confusionMaxtrix(valGT_list, valPre_list, save_csv_path=Test_cfm_path)
    mIoU = mean_Intersection_over_Union(cfm, Test_mIoU_path)
    print('mIoU',mIoU)
    
    fwiou = Frequency_Weighted_mean_Intersection_over_Union(cfm, Test_FWIoU_path)
    print('fwiou',fwiou)

    

def mean_confusionMaxtrix(gts_list, pres_list, save_csv_path):
    '''
    根据gt和pre生成混淆矩阵，并生成可视化的热力图
    :param gts_list:
    :param pres_list:
    :param save_csv_path:
    :param save_pic_path:
    :return:
    '''
    cfM = np.zeros((Label_num, Label_num))
    for index, (gt_path, pre_path) in enumerate(zip(gts_list, pres_list)):
        gt = cv2.imread(gt_path,-1)[512-Img_size//2:512+Img_size//2,1024-Img_size//2:1024+Img_size//2]
        pre = cv2.imread(pre_path,-1)
        try:
            cfM += get_confusionMaxtrix(gt.flatten(), pre.flatten(), Label_num) 
        except:
            continue
        process_show(index + 1, len(gts_list),'cfm:')

    print()
    cn = [cls.name for cls in Clss]

    cfM_df = pd.DataFrame(cfM, index=cn, columns=cn)

    plt.figure(figsize=(20, 20))
    sns.heatmap(cfM_df, annot=True, fmt='.20g', cmap='Blues')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


    plt.savefig(save_csv_path.replace('.csv','.png'))

    cfM_df.loc['Col_sum'] = cfM_df.apply(lambda x: x.sum())
    cfM_df['Row_sum'] = cfM_df.apply(lambda x: x.sum(), axis=1)
    tp = []
    for i in cn:
        tp.append(cfM_df[i].loc[i]/cfM_df['Row_sum'].loc[i]) 
    tp.append('')
    cfM_df['tp'] = tp
    cfM_df.to_csv(save_csv_path)

    return cfM


def get_confusionMaxtrix(label_vector, pre_vector, class_num):
    '''
    单个样本的gt、pre比较，生成混淆矩阵
    :param label_vector:
    :param pre_vector:
    :param class_num:
    :return:
    '''
    mask = (label_vector >= 0) & (label_vector < class_num)
    return np.bincount(class_num * label_vector[mask].astype(int) + pre_vector[mask], minlength=class_num ** 2).reshape(
        class_num, class_num)


def mean_Intersection_over_Union(cfm, save_path):
    '''
    根据混淆矩阵计算mIoU
    :param cfm:
    :param save_path:
    :return:
    '''
    IoU = np.diag(cfm) / (np.sum(cfm, axis=1) + np.sum(cfm, axis=0) - np.diag(cfm))
    mIoU = {}
    for cls in Clss:
        mIoU[cls.name] = IoU[cls.id]
    mIoU['Mean:'] = np.nanmean(IoU)
    mIoU = pd.Series(mIoU)
    mIoU.to_csv(save_path)
    return mIoU

def Frequency_Weighted_mean_Intersection_over_Union(cfm, save_path):
    '''
    根据混淆矩阵计算FWIoU
    :param cfm:
    :param save_path:
    :return:
    '''
    FW = np.sum(cfm,axis=1)/np.sum(cfm)
    IoU = np.diag(cfm) / (np.sum(cfm, axis=1) + np.sum(cfm, axis=0) - np.diag(cfm))
    FWMIoU = (FW[FW > 0] * IoU[FW > 0]).sum()
    return FWMIoU



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
    val_dataset = MyCS_Dataset(imgs_list, None, batch_size=Batch_size, Buffer_size=Batch_size*10, mode='val')
    val_reader = val_dataset.dataset_reader()


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
        for batch_num, samples in enumerate(val_reader()):  
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
                pre_save_path = os.path.join(pres_save_path, file_name+'.png')
                cv2.imwrite(pre_save_path, pres[index])
            process_show(batch_num,val_dataset.batch_num,'test infer:')
        print()

if __name__ == '__main__':
    pass
    test()




