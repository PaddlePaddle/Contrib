import os
from collections import namedtuple


Data_path = r'/home/aistudio/data'
Images_path = os.path.join(Data_path,'leftImg8bit')
Pres_path = os.path.join(Data_path,'pre')
Infer_path = os.path.join(Data_path,'infer')
GTs_path = os.path.join(Data_path,'gtFine')


Results_path = r'/home/aistudio/work/result'
Model_path = os.path.join(Results_path,'danet')

Txt_prepare_path = os.path.join(Results_path, "txt_prepare")  
Split_csv_path = os.path.join(Txt_prepare_path,'split.csv')
Sta_ms_path = os.path.join(Txt_prepare_path, "sta_ms.csv")
Sta_cls_path = os.path.join(Txt_prepare_path, "sta_cls.csv")
Sta_shape_path = os.path.join(Txt_prepare_path, "sta_shape.csv") 
Data_log_path = os.path.join(Txt_prepare_path, 'data_process.log')  

Txt_net_path = os.path.join(Results_path, "txt_net") 
Train_record_path = os.path.join(Txt_net_path, 'train_record.csv')  
Test_cfm_path = os.path.join(Txt_net_path,'cfm_danet.csv')
Test_mIoU_path = os.path.join(Txt_net_path, 'test_mIoU.csv') 
Test_FWIoU_path = os.path.join(Txt_net_path, 'test_FWIoU.csv') 
Train_log_path = os.path.join(Txt_net_path, 'train.log') 
Test_log_path = os.path.join(Txt_net_path, 'test.log') 


Cls = namedtuple('cls', ['name', 'id', 'color', ])
Clss = [
    Cls('0', 0, (0, 200, 0)),
    Cls('1', 1, (150, 250, 0)),
    Cls('2', 2, (150, 200, 150)),
    Cls('3', 3, (200, 0, 200)),
    Cls('4', 4, (150, 0, 250)),
    Cls('5', 5, (150, 150, 250)),
    Cls('6', 6, (250, 200, 0)),
    Cls('7', 7, (25, 25, 112)),
    Cls('8', 8, (150, 150, 250)),
    Cls('9', 9, (250, 200, 0)),
    Cls('10', 10, (25, 25, 112)),
    Cls('11', 11, (150, 250, 0)),
    Cls('12', 12, (150, 200, 150)),
    Cls('13', 13, (200, 0, 200)),
    Cls('14', 14, (150, 0, 250)),
    Cls('15', 15, (150, 150, 250)),
    Cls('16', 16, (250, 200, 0)),
    Cls('17', 17, (25, 25, 112)),
    Cls('18', 18, (150, 150, 250)),
    Cls('19', 19, (250, 200, 0)),
]

Label_num = len(Clss)
Img_size = 768
Img_chs = 3
Means = [0.485*255, 0.456*255, 0.406*255]
Stds = [0.229*255, 0.224*255, 0.225*255]

Reload = True
Use_gpu = True

Train_rate = 0.8
Epoch_num = 30
Batch_size = 8

Learning_rate = 1e-1
LR_piecewise_boundaries = [5, 13, 21]
LR_piecewise_values = [1e-1, 1e-2, 1e-3, 1e-4]
LR_warmup_steps = 2
LR_warmup_start = 1e-4
LR_warmup_end = 1e-1

LR_poly_power = 0.9
LR_poly_min = 1e-4

Momentum=0.9
Decay =  0.0001


def check_dirs(create=True):
    '''
    检查各个文件夹是否存在
    :param create: 对于不存在的文件夹是否自动创建
    :return:
    '''
    dirs = [
        Results_path,
        Txt_prepare_path,
        Txt_net_path,
        Data_path,
        Images_path,
        Pres_path,
        Infer_path,
        GTs_path,
    ]
    for dir_path in dirs:
        if os.path.exists(dir_path):
            print('pass:',dir_path)
        else:
            if create:
                os.makedirs(dir_path)
                print('!!!make:', dir_path)
            else:
                print('!!!no exists:', dir_path)

if __name__ == '__main__':
    pass
    check_dirs(create=True)





















