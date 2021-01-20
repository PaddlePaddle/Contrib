import sys
import logging
import argparse
import ast
import numpy as np
try:
    import cPickle as pickle
except:
    import pickle
import paddle.fluid as fluid

from model import ECO
from reader import KineticsReader
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
        default='eco',
        help='name of model to train.')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/eco.txt',
        help='path to config file of model')
    parser.add_argument(
        '--use_gpu',
        type=ast.literal_eval,
        default=False,
        help='default use gpu.')
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


def test(args):
    config = parse_config(args.config)
    test_config = merge_configs(config, 'test', vars(args))
    # print_configs(test_config, "test")
    with fluid.dygraph.guard():
        test_model = ECO.GoogLeNet(test_config['MODEL']['num_classes'],
                                    test_config['MODEL']['seg_num'], 
                                    test_config['MODEL']['seglen'], 'RGB')

        # get test reader
        test_reader = KineticsReader(args.model_name.upper(), 'test', test_config).create_reader()

        # if no weight files specified, exit()
        if args.weights:
            weights = args.weights
        else:
            print("model path must be specified")
            exit()
            
        para_state_dict, _ = fluid.load_dygraph(weights)
        test_model.load_dict(para_state_dict)
        test_model.eval()
        
        acc_list = []
        for batch_id, data in enumerate(test_reader()):
            dy_x_data = np.array([x[0] for x in data]).astype('float32')
            y_data = np.array([[x[1]] for x in data]).astype('int64')
            
            img = fluid.dygraph.to_variable(dy_x_data)
            label = fluid.dygraph.to_variable(y_data)
            label.stop_gradient = True
            
            out, acc = test_model(img, label)
            acc_list.append(acc.numpy()[0])

        print("The accuracy for test dataset is:{}".format(np.mean(acc_list)))
         
            
if __name__ == "__main__":
    args = parse_args()
    logger.info(args)

    test(args)
