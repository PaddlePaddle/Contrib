import argparse
import pickle
from tqdm import tqdm
import sys

sys.path.extend(['../'])
from data_gen.preprocess import pre_normalization

training_subjects = [
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38
]  # 按照cross_subjects方式进行训练，选取这些id的人的动作作为训练集，其余作为测试集
training_cameras = [2, 3]  # 按照cross_view方式进行训练，选取2,3摄像头的数据为训练集，1摄像头的数据作为测试集
max_body_true = 2  # NTU数据集最多只有两个人
max_body_kinect = 4  # Kinect数据集最多有4个人
num_joint = 25  # 骨骼点数为25个
max_frame = 300  # 最大帧数设置为300，多退少补

import numpy as np
import os


def read_skeleton_filter(file):  # file是文件名S001C001P001R001A001.skeleton
    with open(file, 'r') as f:
        skeleton_sequence = {}
        skeleton_sequence['numFrame'] = int(f.readline())  # 第一行单个样本包含的总帧数
        skeleton_sequence['frameInfo'] = []
        # num_body = 0
        for t in range(skeleton_sequence['numFrame']):  # 对每帧进行读取
            frame_info = {}
            frame_info['numBody'] = int(f.readline())  # 读取每帧中人的数量
            frame_info['bodyInfo'] = []

            for m in range(frame_info['numBody']):  # 读取每个人的信息
                body_info = {}
                body_info_key = [
                    'bodyID', 'clipedEdges', 'handLeftConfidence',
                    'handLeftState', 'handRightConfidence', 'handRightState',
                    'isResticted', 'leanX', 'leanY', 'trackingState'
                ]
                body_info = {
                    k: float(v)
                    for k, v in zip(body_info_key, f.readline().split())
                }  # 读取body_info前面的信息eg:72057594037944738 0 1 1 1 1 0 0.01119184 -0.256052 2
                body_info['numJoint'] = int(f.readline())  # 读取关节点数量
                body_info['jointInfo'] = []
                for v in range(body_info['numJoint']):  # 对每个关节点读取信息
                    joint_info_key = [
                        'x', 'y', 'z', 'depthX', 'depthY', 'colorX', 'colorY',
                        'orientationW', 'orientationX', 'orientationY',
                        'orientationZ', 'trackingState'
                    ]
                    joint_info = {
                        k: float(v)
                        for k, v in zip(joint_info_key, f.readline().split())
                    }
                    body_info['jointInfo'].append(joint_info)
                frame_info['bodyInfo'].append(body_info)
            skeleton_sequence['frameInfo'].append(frame_info)

    return skeleton_sequence


def get_nonzero_std(s):  # tvc
    index = s.sum(-1).sum(-1) != 0  # select valid frames，V和C两维全为0的剔除掉
    s = s[index]
    if len(s) != 0:
        s = s[:, :, 0].std() + s[:, :, 1].std() + s[:, :, 2].std()  # three channels
    else:
        s = 0
    return s  # 返回x,y,z坐标均值的和,代表一帧中每个人的强度，选取强度最大的前M个人


def read_xyz(file, max_body=4, num_joint=25):  # 取了前两个body（ntu中最多只有两个人）
    seq_info = read_skeleton_filter(file)
    data = np.zeros((max_body, seq_info['numFrame'], num_joint, 3))  # 4xTxVx3
    for n, f in enumerate(seq_info['frameInfo']):  # n是帧数
        for m, b in enumerate(f['bodyInfo']):  # m是一帧中的人数
            for j, v in enumerate(b['jointInfo']):  # j是关节数
                if m < max_body and j < num_joint:
                    data[m, n, j, :] = [v['x'], v['y'], v['z']]
                else:
                    pass

    # select two max energy body
    energy = np.array([get_nonzero_std(x) for x in data])
    index = energy.argsort()[::-1][0:max_body_true]  # 取前max_body_true个个体
    data = data[index]  # MxTxVxC

    data = data.transpose(3, 1, 2, 0)  # CxTxVxM
    return data


def gendata(data_path, out_path, ignored_sample_path=None, benchmark='xview', part='eval'):
    if ignored_sample_path != None:
        with open(ignored_sample_path, 'r') as f:
            ignored_samples = [
                line.strip() + '.skeleton' for line in f.readlines()
            ]  # 读取丢失骨骼信息的sample的文件名
    else:
        ignored_samples = []
    sample_name = []
    sample_label = []
    for filename in os.listdir(data_path):  # ../data/nturgbd_raw/nturgb+d_skeletons/
        if filename in ignored_samples:  # 读取的文件属于丢失信息的文件，直接跳过
            continue
        action_class = int(
            filename[filename.find('A') + 1:filename.find('A') + 4])  # action类别，直接输出动作类别
        subject_id = int(
            filename[filename.find('P') + 1:filename.find('P') + 4])  # 输出person序号，用于cross_sub区分训练集和测试集
        camera_id = int(
            filename[filename.find('C') + 1:filename.find('C') + 4])  # 输出camera序号,用于cross_view区分训练集和测试集

        if benchmark == 'xview':
            istraining = (camera_id in training_cameras)  # 判断此sample用于训练还是测试
        elif benchmark == 'xsub':
            istraining = (subject_id in training_subjects)
        else:
            raise ValueError()

        if part == 'train':
            issample = istraining
        elif part == 'val':
            issample = not (istraining)
        else:
            raise ValueError()

        if issample:  # 确定采样此样例
            sample_name.append(filename)
            sample_label.append(action_class - 1)

    with open('{}/{}_label.pkl'.format(out_path, part), 'wb') as f:
        pickle.dump((sample_name, list(sample_label)), f)

    fp = np.zeros((len(sample_label), 3, max_frame, num_joint, max_body_true), dtype=np.float32)  # Nx3xTxVxM

    for i, s in enumerate(tqdm(sample_name)):  # s为文件名，eg:S001C001P001R001A001.skeleton
        data = read_xyz(os.path.join(data_path, s), max_body=max_body_kinect, num_joint=num_joint)
        # 读取sample对应的原始数据文件CxTxVxM
        fp[i, :, 0:data.shape[1], :, :] = data  # 空余的时间帧是0,NxCxTxVxM

    fp = pre_normalization(fp)
    np.save('{}/{}_data_joint.npy'.format(out_path, part), fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NTU-RGB-D Data Converter.')
    parser.add_argument('--data_path', default='../data/nturgbd_raw/nturgb+d_skeletons/')
    parser.add_argument('--ignored_sample_path',
                        default='../data/nturgbd_raw/samples_with_missing_skeletons.txt')
    parser.add_argument('--out_folder', default='../data/ntu/')

    benchmark = ['xsub', 'xview']
    part = ['train', 'val']
    arg = parser.parse_args()

    for b in benchmark:  # 分成两部分，cross_sub和cross_view
        for p in part:  # 拆分训练集和测试集
            out_path = os.path.join(arg.out_folder, b)  # ../data/ntu/(xsub & xview)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            print(b, p)
            gendata(
                arg.data_path,
                out_path,
                arg.ignored_sample_path,  # 这个txt里面包含了所有丢失了部分骨骼点信息的样本，在原始skeleton文件中出现很多0
                benchmark=b,
                part=p)
