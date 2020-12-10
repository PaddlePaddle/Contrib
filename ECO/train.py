import os
import argparse
import ast
import logging
import numpy as np
import paddle.fluid as fluid

from model import ECO
from reader import KineticsReader
from config import parse_config, merge_configs, print_configs

logging.root.handlers = []
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(filename='logger.log', level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser("Paddle Video train script")
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
        type=ast.literal_eval,
        default=False,
        help='path to pretrain weights. None to use default weights path in  ~/.paddle/weights.'
    )
    parser.add_argument(
        '--use_gpu',
        type=ast.literal_eval,
        default=False,
        help='default use gpu.')
    parser.add_argument(
        '--epoch',
        type=int,
        default=10,
        help='epoch number, 0 for read from config file')
    parser.add_argument(
        '--save_dir',
        type=str,
        default='checkpoints_models',
        help='directory name to save train snapshoot')
    args = parser.parse_args()
    return args
    

def validate_model():
    # parse config
    args = parse_args()
    config = parse_config(args.config)
    val_config = merge_configs(config, 'test', vars(args))
    
    val_reader = KineticsReader(args.model_name.upper(), 'test', val_config).create_reader()

    val_model = ECO.GoogLeNet(val_config['MODEL']['num_classes'],
                                    val_config['MODEL']['seg_num'],
                                    val_config['MODEL']['seglen'], 'RGB')

    model, _ = fluid.dygraph.load_dygraph(args.save_dir + '/ucf_model')
    val_model.load_dict(model)
        
    val_model.eval()
    
    acc_list = []
    for batch_id, data in enumerate(val_reader()):
        dy_x_data = np.array([x[0] for x in data]).astype('float32')
        y_data = np.array([[x[1]] for x in data]).astype('int64')
        
        img = fluid.dygraph.to_variable(dy_x_data)
        label = fluid.dygraph.to_variable(y_data)
        label.stop_gradient = True
        
        out, acc = val_model(img, label)
        if out is not None:
            acc_list.append(acc.numpy()[0])

    val_model.train()
    return np.mean(acc_list)

def train(args):
    all_train_rewards=[]
    all_test_rewards=[]
    prev_result=0
    # parse config
    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    with fluid.dygraph.guard(place):
        config = parse_config(args.config)
        train_config = merge_configs(config, 'train', vars(args))
        print_configs(train_config, 'Train')

        train_model = ECO.GoogLeNet(train_config['MODEL']['num_classes'],
                                    train_config['MODEL']['seg_num'],
                                    train_config['MODEL']['seglen'], 'RGB')
        opt = fluid.optimizer.Momentum(0.001, 0.9, parameter_list=train_model.parameters(),use_nesterov=True,regularization=fluid.regularizer.L2Decay(
        regularization_coeff=0.0005))
        
        if args.pretrain:
            model, _ = fluid.dygraph.load_dygraph('trained_model/best_model')
            train_model.load_dict(model)

        # build model
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        # get reader
        train_reader = KineticsReader(args.model_name.upper(), 'train', train_config).create_reader()

        epochs = args.epoch or train_model.epoch_num()

        train_model.train()

        for i in range(epochs):
            for batch_id, data in enumerate(train_reader()):
                dy_x_data = np.array([x[0] for x in data]).astype('float32')
                y_data = np.array([[x[1]] for x in data]).astype('int64')
                
                img = fluid.dygraph.to_variable(dy_x_data)
                label = fluid.dygraph.to_variable(y_data)
                label.stop_gradient = True
                
                out, acc = train_model(img, label)

                if out is not None:
                
                    loss = fluid.layers.cross_entropy(out, label)
                    avg_loss = fluid.layers.mean(loss)

                    avg_loss.backward()

                    opt.minimize(avg_loss)
                    train_model.clear_gradients()
                      
                    if batch_id % 200 == 0:
                        print("Loss at epoch {} step {}: {}, acc: {}".format(i, batch_id, avg_loss.numpy(), acc.numpy()))
                        fluid.dygraph.save_dygraph(train_model.state_dict(), args.save_dir + '/ucf_model')
                        result = validate_model()

                        all_test_rewards.append(result)
                        if result > prev_result:
                            prev_result = result
                            print('The best result is ' + str(result))
                            fluid.save_dygraph(train_model.state_dict(),'trained_model/best_model')
                            np.savez('result_data/ucf_data.npz', all_train_rewards=all_train_rewards, all_test_rewards=all_test_rewards)
                        
            all_train_rewards.append(acc.numpy())
    
        logger.info("Final loss: {}".format(avg_loss.numpy()))
        print("Final loss: {}".format(avg_loss.numpy()))

        np.savez('result_data/ucf_data.npz', all_train_rewards=all_train_rewards, all_test_rewards=all_test_rewards)
            


if __name__ == "__main__":
    args = parse_args()
    # logger.info(args)

    train(args)
