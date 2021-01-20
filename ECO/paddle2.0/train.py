import os
import argparse
import ast
import logging
import numpy as np
import paddle

from model import ECO
from reader import ECO_Dataset
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

    val_dataset = ECO_Dataset(args.model_name.upper(), val_config, mode='test')

    val_loader = paddle.io.DataLoader(val_dataset, places=paddle.CUDAPlace(0), batch_size=None, batch_sampler=None)

    val_model = ECO.GoogLeNet(val_config['MODEL']['num_classes'],
                                    val_config['MODEL']['seg_num'],
                                    val_config['MODEL']['seglen'], 'RGB', 0.00002)
   
    model_dict = paddle.load(args.save_dir + '/ucf_model_hapi')
    val_model.set_state_dict(model_dict)
        
    val_model.eval()
    
    acc_list = []
    for batch_id, data in enumerate(val_loader()):

        img = data[0]
        label = data[1]
        
        out, acc = val_model(img, label)
        if out is not None:
            acc_list.append(acc.numpy()[0])

    val_model.train()
    return np.mean(acc_list)

def train(args):
    all_train_rewards=[]
    all_test_rewards=[]
    prev_result=0

    config = parse_config(args.config)
    train_config = merge_configs(config, 'train', vars(args))
    print_configs(train_config, 'Train')

    train_model = ECO.GoogLeNet(train_config['MODEL']['num_classes'],
                                train_config['MODEL']['seg_num'],
                                train_config['MODEL']['seglen'], 'RGB', 0.00002)
    opt = paddle.optimizer.Momentum(0.001, 0.9, parameters=train_model.parameters())

    if args.pretrain:
        # 加载上一次训练的模型，继续训练
        model_dict = paddle.load('best_model/best_model_seg12')

        train_model.set_state_dict(model_dict)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    train_dataset = ECO_Dataset(args.model_name.upper(), train_config, mode='train')

    train_loader = paddle.io.DataLoader(train_dataset, places=paddle.CUDAPlace(0), batch_size=None, batch_sampler=None)

    epochs = args.epoch or train_model.epoch_num()

    train_model.train()

    for i in range(epochs):

        for batch_id, data in enumerate(train_loader()):

            img = data[0]
            label = data[1]

            out, acc = train_model(img, label)

            if out is not None:
            
                loss = paddle.nn.functional.cross_entropy(out, label)
                avg_loss = paddle.mean(loss)

                avg_loss.backward()

                opt.minimize(avg_loss)
                train_model.clear_gradients()
          
                if batch_id % 200 == 0:
                    print("Loss at epoch {} step {}: {}, acc: {}".format(i, batch_id, avg_loss.numpy(), acc.numpy()))
                    paddle.save(train_model.state_dict(), args.save_dir + '/ucf_model_hapi')
        all_train_rewards.append(acc.numpy())

        result = validate_model()

        all_test_rewards.append(result)
        if result > prev_result:
            prev_result = result
            print('The best result is ' + str(result))
            paddle.save(train_model.state_dict(),'best_model/final_best_model_hapi')#保存模型
    logger.info("Final loss: {}".format(avg_loss.numpy()))
    print("Final loss: {}".format(avg_loss.numpy()))

    np.savez('result/final_ucf_data_hapi.npz', all_train_rewards=all_train_rewards, all_test_rewards=all_test_rewards)

if __name__ == "__main__":
    args = parse_args()
    # logger.info(args)
    train(args)
