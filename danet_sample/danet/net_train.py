import os
import datetime
import numpy as np
import pandas as pd
from paddle import fluid
import matplotlib.pyplot as plt

from dataset.myDataset import *
from networks.danet import *
from my_utils.com_utils import *
from my_utils.global_config import *



def train():
    # 读取数据集
    print('dataset load...')
    df = pd.read_csv(Split_csv_path, header=0, index_col=0)

    trainImg_list = df[df['split'] == 'train']['img_path'].tolist()
    trainGT_list = df[df['split'] == 'train']['gt_path'].tolist()
    train_dataset = MyCS_Dataset(trainImg_list, trainGT_list,
                                batch_size=Batch_size,Buffer_size=Batch_size*5,mode='train')
    train_reader = train_dataset.dataset_reader()
    # LR_piecewise_boundaries_enlarge = [ep*train_dataset.batch_num+1 for ep in LR_piecewise_boundaries]
    # LR_warmup_steps_enlarge = LR_warmup_steps*train_dataset.batch_num+1

    valImg_list = df[df['split'] == 'val']['img_path'].tolist()
    valGT_list = df[df['split'] == 'val']['gt_path'].tolist()
    val_dataset = MyCS_Dataset(valImg_list, valGT_list,
                                batch_size=Batch_size,Buffer_size=Batch_size*5,mode='train')
    val_reader = val_dataset.dataset_reader()
    

    print('model load...')
    place = fluid.CUDAPlace(0) if Use_gpu else fluid.CPUPlace()
    with fluid.dygraph.guard(place):
        model = DANet('danet')

        # LR_scheduler = fluid.layers.piecewise_decay(boundaries=LR_boundaries_enlarge, values=LR_values)
        # LR_warmup = fluid.layers.linear_lr_warmup(LR_scheduler,warmup_steps=LR_warmup_steps_enlarge, start_lr=LR_warmup_start, end_lr=LR_warmup_end)
        # optimizer = fluid.optimizer.AdamOptimizer(learning_rate=LR_warmup, parameter_list=model.parameters())
        criterion = fluid.layers.softmax_with_cross_entropy


        LR_scheduler = fluid.layers.polynomial_decay(Learning_rate, Epoch_num*train_dataset.batch_num, LR_poly_min, power=LR_poly_power)
        l2_decay = fluid.regularizer.L2Decay(regularization_coeff=Decay)
        optimizer = fluid.optimizer.MomentumOptimizer(LR_scheduler, momentum=Momentum, parameter_list=model.parameters(), regularization=l2_decay)


        # 读取上次保存的模型
        if Reload and os.path.exists(Model_path+'.pdparams'):
            print(Model_path,'reload...')
            m_para, m_opt = fluid.load_dygraph(Model_path)
            model.load_dict(m_para)

        print('start train...')

        # 训练过程记录
        record = {
            'train_loss': np.zeros(Epoch_num),
            'train_acc': np.zeros(Epoch_num),
            'val_loss': np.zeros(Epoch_num),
            'val_acc': np.zeros(Epoch_num),
            'lr': np.zeros(Epoch_num),
        }
        best_loss = float("inf")
        best_loss_epoch = 0
        best_acc = 0
        best_acc_epoch = 0


        for epoch in range(Epoch_num):
            print('epoch %d/%d'%(epoch+1,Epoch_num))
            t1 = datetime.datetime.now()
            model.train()
            for batch_num, samples in enumerate(train_reader()):
                # 数据格式转换
                images = np.array([sample[0] for sample in samples])
                gts = np.array([sample[1] for sample in samples])
                images = fluid.dygraph.to_variable(images)
                labels = fluid.dygraph.to_variable(gts)

                # 前向传播
                p_out, c_out, output = model(images)
                # output = model(images)

                # loss计算
                loss_p = criterion(p_out, labels,soft_label=True, axis=1)
                loss_c = criterion(c_out, labels,soft_label=True, axis=1)
                loss_sum = criterion(output, labels,soft_label=True, axis=1)
                loss = 0.3*loss_p+0.3*loss_c+0.4*loss_sum
                # loss = criterion(output, labels,soft_label=True, axis=1)

                loss_mean = fluid.layers.mean(loss)
                loss_mean.backward()
             
                # 反向传播
                optimizer.minimize(loss_mean)
                model.clear_gradients()

                # 精度计算
                pres = np.argmax(output.numpy(), axis=1)
                gts = np.argmax(gts, axis=1)
                correct = (pres == gts).sum().astype('float32')
                acc = correct / (gts.shape[0] * gts.shape[1] * gts.shape[2])

                record['train_loss'][epoch] += loss_mean.numpy()
                record['train_acc'][epoch] += acc
                trainval_show(batch_num + 1, train_dataset.batch_num, acc, loss_mean.numpy(), prefix='train:',suffix=str(optimizer.current_step_lr()))
            print()

            t2 = datetime.datetime.now()
            model.eval()
            for batch_num, samples in enumerate(val_reader()):
                # 数据格式转换
                images = np.array([sample[0] for sample in samples])
                gts = np.array([sample[1] for sample in samples])
                images = fluid.dygraph.to_variable(images)
                labels = fluid.dygraph.to_variable(gts)

                # 前向传播
                p_out, c_out,output = model(images)
                # output = model(images)                

                # loss计算
                loss_p = criterion(p_out, labels,soft_label=True, axis=1)
                loss_c = criterion(c_out, labels,soft_label=True, axis=1)
                loss_sum = criterion(output, labels,soft_label=True, axis=1)
                loss = loss_p+loss_c+loss_sum
                
                # loss = criterion(output, labels,soft_label=True, axis=1)
                loss_mean = fluid.layers.mean(loss)

                # 精度计算                
                pres = np.argmax(output.numpy(), axis=1)
                gts = np.argmax(gts, axis=1)
                correct = (pres == gts).sum().astype('float32')
                acc = correct / (gts.shape[0] * gts.shape[1] * gts.shape[2])

                record['val_loss'][epoch] += loss_mean.numpy()
                record['val_acc'][epoch] += acc
                trainval_show(batch_num + 1, len(val_dataset)//Batch_size, acc, loss_mean.numpy(), prefix='val:',suffix=str(optimizer.current_step_lr()))
            print()
            t3 = datetime.datetime.now()
            print('train time:',t2-t1,' val time:',t3-t2,' system time:',datetime.datetime.now())

            # 单次epoch平均汇总
            record['train_loss'][epoch] /= train_dataset.batch_num
            record['train_acc'][epoch] /= train_dataset.batch_num
            record['val_loss'][epoch] /= val_dataset.batch_num
            record['val_acc'][epoch] /= val_dataset.batch_num
            record['lr'][epoch] = optimizer.current_step_lr()

            # 单次 汇总输出
            print('average summary:\ntrain acc %.4f, loss %.4f ; val acc %.4f, loss %.4f ; lr %.4f'
                % (record['train_acc'][epoch], record['train_loss'][epoch], record['val_acc'][epoch],
                    record['val_loss'][epoch], record['lr'][epoch]))

            # best更新及模型保存
            # best更新及模型保存,优先监督acc提升
            if record['val_acc'][epoch] > best_acc:
                save_path = nameAdd(Model_path)
                print('val_acc improve from %.4f to %.4f, model save to %s ! \n' % (
                    best_acc, record['val_acc'][epoch],save_path))
                best_acc = record['val_acc'][epoch]
                best_acc_epoch = epoch + 1
                fluid.save_dygraph(model.state_dict(),Model_path)
                if record['val_loss'][epoch] < best_loss:
                    best_loss = record['val_loss'][epoch]
                    best_loss_epoch = epoch + 1
            elif record['val_loss'][epoch] < best_loss:
                save_path = nameAdd(Model_path, '_lossBest')
                print('val_loss improve from %.4f to %.4f, model save to %s ! \n' % (
                    best_loss, record['val_loss'][epoch], save_path))
                best_loss = record['val_loss'][epoch]
                best_loss_epoch = epoch + 1
                fluid.save_dygraph(model.state_dict(),save_path)
            else:
                print('No improvement: best loss %.4f at %d; best acc %.4f at %d \n' % (best_loss, best_loss_epoch, best_acc, best_acc_epoch))

        # Record 保存及输出
        print('best loss %.4f at %d; best acc %.4f at %d \n' % (best_loss, best_loss_epoch, best_acc, best_acc_epoch))
        df = pd.DataFrame(record)
        df.to_csv(Train_record_path)
        draw_loss_acc(record, save_path=Train_record_path.replace('.csv','.png'))


def nameAdd(src_path, suf_fix='', pre_fix=''):
    if os.path.isabs(src_path):
        src_name = os.path.basename(src_path)
        src_path = os.path.dirname(src_path)
        filename, extension = os.path.splitext(src_name)
        dst_name = pre_fix + filename + suf_fix + extension
        dst_path = os.path.join(src_path,dst_name)
    else:
        filename, extension = os.path.splitext(src_path)
        dst_path= pre_fix + filename + suf_fix + extension
    return dst_path

def draw_loss_acc(record, save_path=None):
    '''
    绘制每轮train、val过程中的loss、acc
    :param record: loss、acc字典，来自train过程
    :param save_path: 是否保存，如果传入None，不保存
    :return:
    '''
    x = [epoch for epoch in range(len(record['train_loss']))]

    plt.figure(figsize=(12, 9))

    plt.subplot(3, 1, 1)
    plt.plot(x, record['train_acc'], 'b-')
    plt.plot(x, record['val_acc'], 'r--')
    plt.legend(['train_acc', 'val_acc'])
    plt.ylabel('accuracy')

    best_loss = min(record['val_loss'])
    best_acc = max(record['val_acc'])
    plt.title('best loss: %.6f best acc: %.6f ' % (best_loss,best_acc) )

    plt.subplot(3, 1, 2)
    plt.plot(x, record['train_loss'], 'b-')
    plt.plot(x, record['val_loss'], 'r--')
    plt.legend(['train_loss', 'val_loss'])
    plt.ylabel('loss')

    plt.subplot(3, 1, 3)
    plt.plot(x, record['lr'], 'b-')
    plt.ylabel('lr')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    # plt.show()



if __name__ == '__main__':
    pass
    train()






















