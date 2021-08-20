import paddle
import math
import paddle.fluid as fluid
from paddle.fluid.dygraph import Conv2D, Pool2D, BatchNorm, Linear
from paddle.fluid import ParamAttr
import numpy as np


def array2samples_distance(array1, array2):  # input1_pp
    num_point, num_features = array1.shape
    expanded_array1 = fluid.layers.expand(array1, [num_point, 1])
    array2_expand = fluid.layers.unsqueeze(array2, [1])
    expanded_array2 = fluid.layers.expand(array2_expand, [1, num_point, 1])
    expanded_array2_reshaped = fluid.layers.reshape(expanded_array2, [-1, num_features], inplace=False)
    distances = (expanded_array1 - expanded_array2_reshaped) * (expanded_array1 - expanded_array2_reshaped)
    distances = fluid.layers.reduce_sum(distances, dim=1)
    distances_reshaped = fluid.layers.reshape(distances, [num_point, num_point])
    distances = fluid.layers.reduce_min(distances_reshaped, dim=1)
    distances = fluid.layers.mean(distances)
    return distances


# def chamfer_distance_numpy(array1, array2):
#     batch_size, num_point, num_features = array1.shape
#     dist = 0
#     for i in range(batch_size):
#         av_dist1 = array2samples_distance(array1[i], array2[i])
#         av_dist2 = array2samples_distance(array2[i], array1[i])
#         dist = dist + (0.5 * av_dist1 + 0.5 * av_dist2) / batch_size
#     return dist * 100


def chamfer_distance_numpy(array1, array2):
    batch_size, num_point, num_features = array1.shape
    dist = 0
    for i in range(batch_size):
        av_dist1 = array2samples_distance(array1[i], array2[i])
        av_dist2 = array2samples_distance(array2[i], array1[i])
        dist = dist + (av_dist1 + av_dist2) / batch_size
    return dist


def chamfer_distance_numpy_test(array1, array2):
    batch_size, num_point, num_features = array1.shape
    dist_all = 0
    dist1 = 0
    dist2 = 0
    for i in range(batch_size):
        av_dist1 = array2samples_distance(array1[i], array2[i])
        av_dist2 = array2samples_distance(array2[i], array1[i])
        dist_all = dist_all + (av_dist1 + av_dist2) / batch_size
        dist1 = dist1 + av_dist1 / batch_size
        dist2 = dist2 + av_dist2 / batch_size
    return dist_all, dist1, dist2


class PointLoss(fluid.dygraph.Layer):
    def __init__(self):
        super(PointLoss, self).__init__()

    def forward(self, array1, array2):
        return chamfer_distance_numpy(array1, array2)


class PointLoss_test(fluid.dygraph.Layer):
    def __init__(self):
        super(PointLoss_test, self).__init__()

    def forward(self, array1, array2):
        return chamfer_distance_numpy_test(array1, array2)


def distance_squre(p1, p2):
    return np.linalg.norm(p1-p2)


def index_points_numpy(points, idx):
    B = points.shape[0]
    npoints = idx.shape[1]
    new_points = np.zeros((B, npoints, 3))
    batch_indices = np.arange(start=0, stop=B)
    for i in batch_indices:
        new_points[i, :, :] = points[i, idx[i, :], :]

    return new_points.astype('float32')


def farthest_point_sample_numpy(xyz, npoint, RAN=True):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    B, N, C = xyz.shape
    centroids = np.zeros((B, npoint))
    distance = np.ones((B, N)) * 1e10
    if RAN:
        farthest = np.random.randint(low=0, high=1, size=(B,))
    else:
        farthest = np.random.randint(low=1, high=2, size=(B,))
    batch_indices = np.arange(start=0, stop=B)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view().reshape(B, 1, 3)
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    return centroids.astype('int64')
