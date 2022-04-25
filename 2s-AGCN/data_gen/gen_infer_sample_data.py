import os.path
import os
import argparse
import numpy as np
import pickle


def get_args(add_help=True):
    """
    parse args
    """
    parser = argparse.ArgumentParser(
        description="gen sample data", add_help=add_help)

    parser.add_argument(
        '--save-dir',
        type=str,
        default='./data/ntu/tiny_dataset',
        help='save path of result data')
    parser.add_argument(
        '--data-num',
        type=int,
        default=50,
        help='data num of result data')
    
    parser.add_argument(
        '--dataset',
        type=str,
        default="xview",
        help='xview or xsub')

    parser.add_argument(
        '--mode',
        type=str,
        default="joint",
        help='joint or bone')

    args = parser.parse_args()
    return args


def gen_tiny_data(data_path, label_path, save_dir, data_num, use_mmap=True):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    if use_mmap:
        data = np.load(data_path, mmap_mode='r')
    else:
        data = np.load(data_path)
    try:
        with open(label_path) as f:
            sample_name, label = pickle.load(f)
    except:
        # for pickle file from python2
        with open(label_path, 'rb') as f:
            sample_name, label = pickle.load(f, encoding='latin1')

    label = label[0:data_num]
    data = data[0:data_num]
    sample_name = sample_name[0:data_num]

    with open(os.path.join(save_dir, "tiny_infer_label.pkl"), 'wb') as f:  # 生成小数据
        pickle.dump((sample_name, list(label)), f)
    np.save(os.path.join(save_dir, "tiny_infer_data"), data)
    print("Successfully generate tiny dataset")


if __name__ == "__main__":
    args = get_args()
    data_path = f'./data/ntu/{args.dataset}/val_data_{args.mode}.npy'
    label_path = f'./data/ntu/{args.dataset}/val_label.pkl'
    gen_tiny_data(data_path, label_path,save_dir=args.save_dir, data_num=args.data_num)

    """
    try:
        with open('../data/ntu/tiny_dataset/tiny_infer_label.pkl') as f:
            sample_name, label = pickle.load(f)
    except:
        # for pickle file from python2
        with open('../data/ntu/tiny_dataset/tiny_infer_label.pkl', 'rb') as f:
            sample_name, label = pickle.load(f, encoding='latin1')
    print(label[0])
    """


