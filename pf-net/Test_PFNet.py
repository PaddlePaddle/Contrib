import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
# import open3d as o3d
import numpy as np
import random
import paddle.fluid as fluid
import argparse
from shapenet_part_loader import PartDataset
import utils
from utils import distance_squre, PointLoss
import copy
from model_PFNet import PFNetG

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot',  default='dataset/train', help='path to dataset')
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--pnum', type=int, default=2048, help='the point number of a sample')
parser.add_argument('--crop_point_num', type=int, default=512, help='0 means do not use else use with this weight')
parser.add_argument('--nc', type=int, default=3)
parser.add_argument('--niter', type=int, default=201, help='number of epochs to train for')
parser.add_argument('--weight_decay', type=float, default=0.001)
parser.add_argument('--learning_rate', default=0.0002, type=float, help='learning rate in training')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
parser.add_argument('--cuda', type=bool, default=False, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=2, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--drop', type=float, default=0.2)
parser.add_argument('--num_scales', type=int, default=3, help='number of scales')
parser.add_argument('--point_scales_list', type=list, default=[2048, 1024, 512], help='number of points in each scales')
parser.add_argument('--each_scales_size', type=int, default=1, help='each scales size')
parser.add_argument('--wtl2', type=float, default=0.95, help='0 means do not use else use with this weight')
parser.add_argument('--cropmethod', default='random_center', help='random|center|random_center')
opt = parser.parse_args()

dset = PartDataset(
    root='/home/arclab/PF-Net-Point-Fractal-Network/dataset/shapenet_part/shapenetcore_partanno_segmentation_benchmark_v0/',
    classification=True, class_choice=None, num_point=opt.pnum, mode='test')

crop_choice = [np.array([1, 0, 0]), np.array([0, 0, 1]), np.array([1, 0, 1]), np.array([-1, 0, 0]), np.array([-1, 1, 0])]
place = fluid.CUDAPlace(0)  # 或者 fluid.CUDAPlace(0)

with fluid.dygraph.guard(place):
    netG = PFNetG(opt.num_scales, opt.each_scales_size, opt.point_scales_list, opt.crop_point_num)
    # netG_scheduler = fluid.dygraph.StepDecay(0.0001, step_size=40, decay_rate=0.2)
    netG_optimizer = fluid.optimizer.AdamOptimizer(learning_rate=0.0001, epsilon=1e-05,
                                                parameter_list=netG.parameters(),
                                                regularization=fluid.regularizer.L2Decay(regularization_coeff=
                                                                                            opt.weight_decay))

    para, netG_opt = fluid.load_dygraph('Checkpoints/netG_pretrained.pdparams')
    netG.load_dict(para)
    netG.eval()

    criterion_G = PointLoss()
    train_loader = fluid.io.DataLoader.from_generator(capacity=10, iterable=True)
    train_loader.set_sample_list_generator(dset.get_reader(opt.batchSize), places=place)


    alpha1 = 0.01
    alpha2 = 0.02

    for data in train_loader():
        points, label = data
        batch_size = points.shape[0]

        real_point = points.numpy()
        real_center = np.zeros((batch_size, opt.crop_point_num, 3)).astype('float32')
        cropped_point = copy.deepcopy(real_point)

        for m in range(batch_size):
            index = random.sample(crop_choice, 1)
            distance_list = []
            p_center = index[0]

            for n in range(opt.pnum):
                distance_list.append(distance_squre(real_point[m, n], p_center))
            distance_order = sorted(enumerate(distance_list), key=lambda x: x[1])

            for sp in range(opt.crop_point_num):
                cropped_point[m, distance_order[sp][0]] = np.array([0, 0, 0])
                real_center[m, sp] = real_point[m, distance_order[sp][0]]

        cropped_point1_idx = utils.farthest_point_sample_numpy(cropped_point, opt.point_scales_list[1], RAN=True)
        cropped_point1 = utils.index_points_numpy(cropped_point, cropped_point1_idx)

        cropped_point2_idx = utils.farthest_point_sample_numpy(cropped_point, opt.point_scales_list[2], RAN=False)
        cropped_point2 = utils.index_points_numpy(cropped_point, cropped_point2_idx)

        cropped_point = fluid.dygraph.to_variable(cropped_point)
        cropped_point1 = fluid.dygraph.to_variable(cropped_point1)
        cropped_point2 = fluid.dygraph.to_variable(cropped_point2)

        real_center1_idx = utils.farthest_point_sample_numpy(real_center, 64, RAN=False)
        real_center1 = utils.index_points_numpy(real_center, real_center1_idx)

        real_center2_idx = utils.farthest_point_sample_numpy(real_center, 128, RAN=True)
        real_center2 = utils.index_points_numpy(real_center, real_center2_idx)

        real_center = fluid.dygraph.to_variable(real_center)
        real_center1 = fluid.dygraph.to_variable(real_center1)
        real_center2 = fluid.dygraph.to_variable(real_center2)

        # cropped_point = np.load('cmp/input_cropped1.npy')
        # cropped_point1 = np.load('cmp/input_cropped2.npy')
        # cropped_point2 = np.load('cmp/input_cropped3.npy')
        #
        # cropped_point = fluid.dygraph.to_variable(cropped_point)
        # cropped_point1 = fluid.dygraph.to_variable(cropped_point1)
        # cropped_point2 = fluid.dygraph.to_variable(cropped_point2)

        cropped_input = [cropped_point, cropped_point1, cropped_point2]

        fake_center1, fake_center2, fake = netG(cropped_input)

        # fake_torch = np.squeeze(np.load('cmp/fake.npy'), 1)
        # fake_center1_torch = np.load('cmp/fake_center1.npy')
        # fake_center2_torch = np.load('cmp/fake_center2.npy')
        #
        # real_center = np.load('cmp/real_center.npy')
        # real_center1 = np.load('cmp/real_center_key1.npy')
        # real_center2 = np.load('cmp/real_center_key2.npy')
        #
        # real_center = fluid.dygraph.to_variable(real_center)
        # real_center1 = fluid.dygraph.to_variable(real_center1)
        # real_center2 = fluid.dygraph.to_variable(real_center2)

        cd_loss = criterion_G(fake, real_center)
        # CD_LOSS_torch = np.load('cmp/CD_LOSS.npy')
        # print(CD_LOSS_torch)
        # print(cd_loss)

        G_loss_l2 = criterion_G(fake, real_center) + alpha1*criterion_G(fake_center1, real_center1) + \
                    alpha2*criterion_G(fake_center2, real_center2)
        print('G_loss: ', G_loss_l2.numpy())

        # real_pc = o3d.geometry.PointCloud()
        # real_pc.points = o3d.utility.Vector3dVector(points[0].numpy())

        # cropped_pc = o3d.geometry.PointCloud()
        # cropped_pc.points = o3d.utility.Vector3dVector(cropped_point[0].numpy())
        # cropped_pc.paint_uniform_color([1, 0.706, 0])

        # fake_pc = o3d.geometry.PointCloud()
        # fake_pc.points = o3d.utility.Vector3dVector(real_center[0].numpy())
        # fake_pc.paint_uniform_color([1, 0.203, 0])
        # o3d.visualization.draw_geometries([cropped_pc, fake_pc])

