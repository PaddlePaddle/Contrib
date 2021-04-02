import os
import sys
import time
import argparse
import ast
import logging
import numpy as np
import paddle.fluid as fluid
from collections import OrderedDict
import zipfile
import datetime
import glob

from model.TSN3D import TSN3D, r50f8s8, r50f32s2
from reader import KineticsReader

from config import parse_config, merge_configs, print_configs

output_dir = "/root/paddlejob/workspace/output"
dest_dir = "/root/paddlejob/workspace/datasets/"
datasets_prefix = '/root/paddlejob/workspace/train_data/datasets/'
train_datasets = datasets_prefix + 'data49371/k400.zip'

logging.root.handlers = []
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(filename=os.path.join(output_dir, 'logger.log'), level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


class ZFile(object):
    """
    文件压缩
    """

    def zip_file(self, fs_name, fz_name):
        """
        从压缩文件
        :param fs_name: 源文件名
        :param fz_name: 压缩后文件名
        :return:
        """
        flag = False
        if fs_name and fz_name:
            try:
                with zipfile.ZipFile(fz_name, mode='w', compression=zipfile.ZIP_DEFLATED) as zipf:
                    zipf.write(fs_name)
                    print(
                        "%s is running [%s] " %
                        (currentThread().getName(), fs_name))
                    print('压缩文件[{}]成功'.format(fs_name))
                if zipfile.is_zipfile(fz_name):
                    os.remove(fs_name)
                    print('删除文件[{}]成功'.format(fs_name))
                flag = True
            except Exception as e:
                print('压缩文件[{}]失败'.format(fs_name), str(e))

        else:
            print('文件名不能为空')
        return {'file_name': fs_name, 'flag': flag}

    def unzip_file(self, fz_name, path):
        """
        解压缩文件
        :param fz_name: zip文件
        :param path: 解压缩路径
        :return:
        """
        flag = False

        if zipfile.is_zipfile(fz_name):  # 检查是否为zip文件
            with zipfile.ZipFile(fz_name, 'r') as zipf:
                zipf.extractall(path)
                flag = True

        return {'file_name': fz_name, 'flag': flag}


def parse_args():
    parser = argparse.ArgumentParser("Paddle Video train script")
    parser.add_argument(
        '--model_name',
        type=str,
        default='tpn',
        help='name of model to train.')
    parser.add_argument(
        '--config',
        type=str,
        default='./configs/tpn_kinetics400.txt',
        help='path to config file of model')
    parser.add_argument(
        '--log_interval',
        type=int,
        default=200,
        help='mini-batch interval to log.')
    parser.add_argument(
        '--valid_interval',
        type=int,
        default=1,
        help='valid interval to log.')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='training batch size. None to use config file setting.')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=None,
        help='learning rate use for training. None to use config file setting.')
    parser.add_argument(
        '--pretrain',
        type=str,
        default=output_dir + '/resnet50.pdparams',
        help='path to pretrain weights. None to use default weights path in  ~/.paddle/weights.'
    )
    parser.add_argument(
        '--resume',
        type=ast.literal_eval,
        default=True,
        help=''
    )
    parser.add_argument(
        '--use_gpu',
        type=ast.literal_eval,
        default=True,
        help='default use gpu.')
    parser.add_argument(
        '--epoch',
        type=int,
        default=None,
        help='epoch number, 0 for read from config file')
    parser.add_argument(
        '--save_dir',
        type=str,
        default=output_dir + '/checkpoints',
        help='directory name to save train snapshoot')
    args = parser.parse_args()
    return args


# 连续保存三次
def save_model(prog, path):
    for i in range(3):
        filelist = glob.glob(path + '/*')
        print(filelist)
        fluid.save(prog, path + '/tpn')
        size = os.path.getsize(path + '/tpn.pdparams')
        if (size > 550000000):
            print('save {} success!'.format(i + 1))
            os.remove(path + '/tpn.pdmodel')
            os.remove(path + '/tpn.pdopt')
            break
        else:
            print('save {} fail!'.format(i + 1))


def train(args):
    # parse config
    config = parse_config(args.config)
    train_config = merge_configs(config, 'train', vars(args))
    valid_config = merge_configs(config, 'valid', vars(args))
    print_configs(train_config, 'Train')

    startup = fluid.Program()
    train_prog = fluid.Program()
    model_para = r50f8s8
    with fluid.program_guard(train_prog, startup):
        with fluid.unique_name.guard():
            #        input = fluid.data(name='image', shape=(None,1,24,224,224), dtype='float32')
            input = fluid.data(name='image', shape=(
            None, 1, model_para['sample']['seglen'] * 3, train_config.TRAIN.target_size,
            train_config.TRAIN.target_size), dtype='float32')
            label = fluid.data(name='label', shape=(None, 1), dtype='int64')
            model_train = TSN3D(config=model_para, is_training=True)
            train_fetch_list = model_train.net(input, label)
            opt = fluid.optimizer.Momentum(config.TRAIN.learning_rate, 0.9, use_nesterov=True,
                                           regularization=fluid.regularizer.L2Decay(config.TRAIN.l2_weight_decay))
            loss = train_fetch_list[2] + train_fetch_list[3]
            opt.minimize(loss)

    valid_prog = fluid.Program()
    with fluid.program_guard(valid_prog, startup):
        with fluid.unique_name.guard():
            #      input = fluid.data(name='image', shape=(None, 1, 24, 224, 224), dtype='float32')
            input = fluid.data(name='image', shape=(
            None, 1, model_para['sample']['seglen'] * 3, valid_config.VALID.target_size,
            valid_config.VALID.target_size), dtype='float32')
            label = fluid.data(name='label', shape=(None, 1), dtype='int64')
            model_valid = TSN3D(config=model_para, is_training=False)
            vaild_fetch_list = model_valid.net(input, label)

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup)

    # build model
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # 加载预训练参数
    if args.resume == True:
        # os.system('cat ./checkpoints/xa* > /root/paddlejob/workspace/output/tpn.pdparams')
        # state_dict = fluid.load_program_state(output_dir+'/tpn.pdparams')
        os.system('cat ./checkpoints/xa* > ' + args.save_dir + '/tpn.pdparams')
        #    state_dict = fluid.load_program_state(output_dir+'/tpn.pdparams')
        state_dict = fluid.load_program_state(args.save_dir + '/tpn.pdparams')
        fluid.set_program_state(train_prog, state_dict)
        #   os.system('rm -f '+ output_dir+'/tpn.pdparams')
        #   print('Resueme from ' + output_dir + '/tpn.pdparams')
        print('Resueme from ' + args.save_dir + '/tpn.pdparams')

    # 用2D参数初始化
    elif args.pretrain:
        os.system('cat ./pretrain/resnet_50_pretrained/xa* > /root/paddlejob/workspace/output/resnet50.pdparams')
        state_dict = fluid.load_program_state(args.pretrain)
        para = train_prog.all_parameters()
        dict_keys = list(state_dict.keys())
        # 膨胀2D参数
        for i in range(len(para)):
            var_name = para[i].name
            if (var_name in dict_keys):
                if (('weight' in var_name) and ('bn' not in var_name)):
                    tmp = state_dict[var_name]
                    para_shape = para[i].shape
                    tmp = tmp.reshape([para_shape[0], para_shape[1], 1, para_shape[3], para_shape[4]])
                    if (para_shape[-3] == 3):
                        tmp = tmp.repeat(3, axis=2)
                        tmp = tmp / 3
                    state_dict[var_name] = tmp
                    print('Inflated {} from pretrained parameters.'.format(var_name))
        # 删除fc参数
        for name in dict_keys:
            if "fc" in name:
                del state_dict[name]
                print('Delete {} from pretrained parameters. Do not load it'.format(name))

        fluid.set_program_state(train_prog, state_dict)
        os.system('rm ' + args.pretrain)
    else:
        pass;

    # 设置多卡运行环境
    build_strategy = fluid.BuildStrategy()
    build_strategy.enable_inplace = True
    build_strategy.sync_batch_norm = True
    exec_strategy = fluid.ExecutionStrategy()

    compiled_train_prog = fluid.compiler.CompiledProgram(
        train_prog).with_data_parallel(
        loss_name=loss.name,
        build_strategy=build_strategy,
        exec_strategy=exec_strategy)
    compiled_valid_prog = fluid.compiler.CompiledProgram(
        valid_prog).with_data_parallel(
        share_vars_from=compiled_train_prog,
        build_strategy=build_strategy,
        exec_strategy=exec_strategy)

    # 获取GPU数量,设置并行reader
    bs_denominator = 1

    print('gpus_num={}'.format(bs_denominator))
    if args.use_gpu:
        # check number of GPUs
        gpus = os.getenv("CUDA_VISIBLE_DEVICES", "")
        print(gpus)
        if gpus == "":
            pass
        else:
            gpus = gpus.split(",")
            num_gpus = len(gpus)
            print(num_gpus)
            assert num_gpus == train_config.TRAIN.num_gpus, \
                "num_gpus({}) set by CUDA_VISIBLE_DEVICES " \
                "shoud be the same as that " \
                "set in {}({})".format(
                    num_gpus, args.config, train_config.TRAIN.num_gpus)
        bs_denominator = train_config.TRAIN.num_gpus
    print('bs_denominator={}'.format(bs_denominator))
    exe_places = fluid.cuda_places()
    # get reader
    train_config.TRAIN.batch_size = int(train_config.TRAIN.batch_size / bs_denominator)
    train_config.MODEL.seglen = model_para['sample']['seglen']
    train_config.MODEL.step = model_para['sample']['step']
    valid_config.VALID.batch_size = int(valid_config.VALID.batch_size / bs_denominator)
    #   train_reader = Ucf101Reader(args.model_name.upper(), 'train', train_config).create_reader()
    train_reader = KineticsReader(args.model_name.upper(), 'train', train_config).create_reader()
    train_dataloader = fluid.io.DataLoader.from_generator(feed_list=[input, label], capacity=16, iterable=True)
    train_dataloader.set_sample_list_generator(train_reader, places=exe_places)

    valid_config.VALID.batch_size = valid_config.VALID.batch_size
    valid_reader = KineticsReader(args.model_name.upper(), 'valid', valid_config).create_reader()
    valid_dataloader = fluid.io.DataLoader.from_generator(feed_list=[input, label], capacity=16, iterable=True)
    valid_dataloader.set_sample_list_generator(valid_reader, places=exe_places)

    epochs = train_config.TRAIN.epoch
    log_interval = args.log_interval
    valid_interval = args.valid_interval

    for i in range(epochs):
        log_lr_and_step()
        print(datetime.datetime.now())
        logger.info(datetime.datetime.now())
        cur_time = 0
        for batchid, data in enumerate(train_dataloader()):
            period = time.time() - cur_time
            cur_time = time.time()
            cls_score, acc, aux_loss, cls_loss = exe.run(compiled_train_prog, fetch_list=train_fetch_list,
                                                         feed=data)
            acc = np.mean(acc)
            aux_loss = np.mean(aux_loss)
            cls_loss = np.mean(cls_loss)

            if (batchid > 0 and (batchid % log_interval == 0)):
                print('Epoch {} iter {} : acc: {}  aux_loss:{} cls_loss: {} period:{}'.format(
                    i, batchid, acc, aux_loss, cls_loss, period))
                logger.error('Epoch {} iter {} : acc: {}  aux_loss:{} cls_loss: {} period:{}'.format(
                    i, batchid, acc, aux_loss, cls_loss, period))
        #   fluid.save(train_prog,args.save_dir+'/tpn')
        save_model(train_prog, args.save_dir)

        if ((i % valid_interval) == 0):
            acc_all = []
            aux_loss_all = []
            cls_loss_all = []
            datetime.datetime.now()
            for batchid, data in enumerate(valid_dataloader()):
                _, acc, aux_loss, cls_loss = exe.run(compiled_valid_prog, fetch_list=vaild_fetch_list, feed=data)
                acc = np.mean(acc)
                aux_loss = np.mean(aux_loss)
                cls_loss = np.mean(cls_loss)
                acc_all.append(acc)
                aux_loss_all.append(aux_loss)
                cls_loss_all.append(cls_loss)
                if (batchid > 0 and (batchid % 50 == 0)):
                    print('Valid iter {} : acc: {}  aux_loss:{} cls_loss: {}'.format(
                        batchid, acc, aux_loss, cls_loss))
                    logger.info('Valid iter {} : acc: {}  aux_loss:{} cls_loss: {}'.format(
                        batchid, acc, aux_loss, cls_loss))
            acc = np.mean(acc_all)
            aux_loss = np.mean(aux_loss_all)
            cls_loss = np.mean(cls_loss_all)
            print('Valid: acc: {}  aux_loss:{} cls_loss: {}'.format(acc, aux_loss, cls_loss))
            logger.info('Valid: acc: {}  aux_loss:{} cls_loss: {}'.format(acc, aux_loss, cls_loss))


def log_lr_and_step():
    try:
        # In optimizers, if learning_rate is set as constant, lr_var
        # name is 'learning_rate_0', and iteration counter is not
        # recorded. If learning_rate is set as decayed values from
        # learning_rate_scheduler, lr_var name is 'learning_rate',
        # and iteration counter is recorded with name '@LR_DECAY_COUNTER@',
        # better impliment is required here
        lr_var = fluid.global_scope().find_var("learning_rate")
        if not lr_var:
            lr_var = fluid.global_scope().find_var("learning_rate_0")
        lr = np.array(lr_var.get_tensor())

        lr_count = '[-]'
        lr_count_var = fluid.global_scope().find_var("@LR_DECAY_COUNTER@")
        if lr_count_var:
            lr_count = np.array(lr_count_var.get_tensor())
        logger.info("------- learning rate {}, learning rate counter {} -----"
                    .format(np.array(lr), np.array(lr_count)))
        print("------- learning rate {}, learning rate counter {} -----"
              .format(np.array(lr), np.array(lr_count)))
    except:
        logger.info("Unable to get learning_rate and LR_DECAY_COUNTER.")
        print("------- learning rate {}, learning rate counter {} -----"
              .format(np.array(lr), np.array(lr_count)))


if __name__ == "__main__":
    # os.system('export CUDA_VISIBLE_DEVICES=0,1,2,3')
    os.system("mkdir " + dest_dir)  # #解压数据集
    if zipfile.is_zipfile(train_datasets):  # 检查是否为zip文件
        with zipfile.ZipFile(train_datasets, 'r') as zipf:
            zipf.extractall(dest_dir)
        print('unzip success.')

    args = parse_args()
    # check whether the installed paddle is compiled with GPU
    logger.info(args)
    print(datetime.datetime.now())
    logger.info(datetime.datetime.now())
    train(args)
