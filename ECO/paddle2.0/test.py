import sys
import logging
import argparse
import numpy as np
try:
    import cPickle as pickle
except:
    import pickle

import paddle

from model import ECO
from reader import ECO_Dataset
from config import parse_config, merge_configs, print_configs

logging.root.handlers = []
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.DEBUG, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_name',
        type=str,
        default='tsn',
        help='name of model to train.')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/eco.txt',
        help='path to config file of model')
    parser.add_argument(
        '--weights',
        type=str,
        default=None,
        help='weight path, None to use weights from Paddle.')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='sample number in a batch for inference.')
    parser.add_argument(
        '--filelist',
        type=str,
        default=None,
        help='path to inferenece data file lists file.')
    parser.add_argument(
        '--log_interval',
        type=int,
        default=1,
        help='mini-batch interval to log.')
    parser.add_argument(
        '--infer_topk',
        type=int,
        default=1,
        help='topk predictions to restore.')
    parser.add_argument(
        '--save_dir', type=str, default='./output', help='directory to store results')
    args = parser.parse_args()
    return args


def eval(args):
    # parse config
    config = parse_config(args.config)
    val_config = merge_configs(config, 'test', vars(args))
    # print_configs(val_config, "test")

    val_model = ECO.GoogLeNet(val_config['MODEL']['num_classes'],
                                val_config['MODEL']['seg_num'], 
                                val_config['MODEL']['seglen'], 'RGB')

    label_dic = np.load('label_dir.npy', allow_pickle=True).item()
    label_dic = {v: k for k, v in label_dic.items()}

    val_dataset = ECO_Dataset(args.model_name.upper(), val_config, mode='test')

    val_loader = paddle.io.DataLoader(val_dataset, places=paddle.CUDAPlace(0), batch_size=None, batch_sampler=None)

    if args.weights:
        weights = args.weights
    else:
        print("model path must be specified")
        exit()
        
    para_state_dict = paddle.load(weights)
    val_model.set_state_dict(para_state_dict)
    val_model.eval()
    
    acc_list = []
    for batch_id, data in enumerate(val_loader()):
        img = data[0]
        label = data[1]
        
        out, acc = val_model(img, label)
        acc_list.append(acc.numpy()[0])

    print("测试集准确率为:{}".format(np.mean(acc_list)))
                     
            
if __name__ == "__main__":
    args = parse_args()
    logger.info(args)

    eval(args)
