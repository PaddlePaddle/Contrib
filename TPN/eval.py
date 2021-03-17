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
        default=2,
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


def eval(args):
    # parse config
    config = parse_config(args.config)
    test_config = merge_configs(config, 'test', vars(args))

    startup = fluid.Program()
    test_prog = fluid.Program()
    model_para = r50f8s8

    with fluid.program_guard(test_prog, startup):
        with fluid.unique_name.guard():
            input = fluid.data(name='image', shape=(
            None, test_config.TEST.seg_num * 3, model_para['sample']['seglen'] * 3, test_config.TEST.target_size,
            test_config.TEST.target_size), dtype='float32')
            label = fluid.data(name='label', shape=(None, 1), dtype='int64')
            model_test = TSN3D(config=model_para, is_training=False)
            test_fetch_list = model_test.net(input, label)

    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(startup)

    # 加载训练好的模型
    os.system('cat ./checkpoints/xa* > /root/paddlejob/workspace/output/tpn.pdparams')
    state_dict = fluid.load_program_state(output_dir + '/tpn.pdparams')
    fluid.set_program_state(test_prog, state_dict)
    print('test: ' + output_dir + '/tpn.pdparams')
    os.system('rm -f ' + output_dir + '/tpn.pdparams')

    # 设置多卡运行环境
    build_strategy = fluid.BuildStrategy()
    build_strategy.enable_inplace = True
    exec_strategy = fluid.ExecutionStrategy()

    compiled_test_prog = fluid.compiler.CompiledProgram(
        test_prog).with_data_parallel(
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
            assert num_gpus == test_config.TRAIN.num_gpus, \
                "num_gpus({}) set by CUDA_VISIBLE_DEVICES " \
                "shoud be the same as that " \
                "set in {}({})".format(
                    num_gpus, args.config, test_config.TRAIN.num_gpus)
        bs_denominator = test_config.TRAIN.num_gpus
    print('bs_denominator={}'.format(bs_denominator))
    exe_places = fluid.cuda_places()
    # get reader
    test_config.TEST.batch_size = int(test_config.TEST.batch_size / bs_denominator)
    test_config.MODEL.seglen = model_para['sample']['seglen']
    test_config.MODEL.step = model_para['sample']['step']

    test_reader = KineticsReader(args.model_name.upper(), 'test', test_config).create_reader()
    test_dataloader = fluid.io.DataLoader.from_generator(feed_list=[input, label], capacity=16, iterable=True)
    test_dataloader.set_sample_list_generator(test_reader, places=exe_places)

    logger.info(datetime.datetime.now())
    print(datetime.datetime.now())

    acc_all = []
    aux_loss_all = []
    cls_loss_all = []
    datetime.datetime.now()
    for batchid, data in enumerate(test_dataloader()):
        _, acc, aux_loss, cls_loss = exe.run(compiled_test_prog, fetch_list=test_fetch_list, feed=data)
        acc = np.mean(acc)
        aux_loss = np.mean(aux_loss)
        cls_loss = np.mean(cls_loss)
        acc_all.append(acc)
        aux_loss_all.append(aux_loss)
        cls_loss_all.append(cls_loss)
        if (batchid > 0 and (batchid % 20 == 0)):
            print('Tset iter {} : acc: {}  aux_loss:{} cls_loss: {}'.format(
                batchid, acc, aux_loss, cls_loss))
            logger.info('Test iter {} : acc: {}  aux_loss:{} cls_loss: {}'.format(
                batchid, acc, aux_loss, cls_loss))
    acc = np.mean(acc_all)
    aux_loss = np.mean(aux_loss_all)
    cls_loss = np.mean(cls_loss_all)
    print('Test: acc: {}  aux_loss:{} cls_loss: {}'.format(acc, aux_loss, cls_loss))
    logger.info('Test: acc: {}  aux_loss:{} cls_loss: {}'.format(acc, aux_loss, cls_loss))


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
    eval(args)
