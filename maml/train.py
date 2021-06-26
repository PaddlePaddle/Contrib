#encoding=utf8
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
import pickle
from MAML import *
import argparse


def load_data_cache(dataset, args):
    n_way = args.n_way
    k_spt = args.k_spt  # support data 的个数
    k_query = args.k_query  # query data 的个数
    imgsz = 28
    resize = imgsz
    task_num = args.task_num
    batch_size = task_num
    #  take 5 way 1 shot as example: 5 * 1
    setsz = k_spt * n_way
    querysz = k_query * n_way
    data_cache = []

    # print('preload next 10 caches of batch_size of batch.')
    for sample in range(50):  # num of epochs

        x_spts, y_spts, x_qrys, y_qrys = [], [], [], []
        for i in range(batch_size):  # one batch means one set

            x_spt, y_spt, x_qry, y_qry = [], [], [], []
            selected_cls = np.random.choice(dataset.shape[0], n_way, replace=False)

            for j, cur_class in enumerate(selected_cls):
                selected_img = np.random.choice(20, k_spt + k_query, replace=False)

                # 构造support集和query集
                x_spt.append(dataset[cur_class][selected_img[:k_spt]])
                x_qry.append(dataset[cur_class][selected_img[k_spt:]])
                y_spt.append([j for _ in range(k_spt)])
                y_qry.append([j for _ in range(k_query)])

            # shuffle inside a batch
            perm = np.random.permutation(n_way * k_spt)
            x_spt = np.array(x_spt).reshape(n_way * k_spt, 1, resize, resize)[perm]
            y_spt = np.array(y_spt).reshape(n_way * k_spt)[perm]
            perm = np.random.permutation(n_way * k_query)
            x_qry = np.array(x_qry).reshape(n_way * k_query, 1, resize, resize)[perm]
            y_qry = np.array(y_qry).reshape(n_way * k_query)[perm]

            # append [sptsz, 1, 84, 84] => [batch_size, setsz, 1, 84, 84]
            x_spts.append(x_spt)
            y_spts.append(y_spt)
            x_qrys.append(x_qry)
            y_qrys.append(y_qry)

        #         print(x_spts[0].shape)
        # [b, setsz = n_way * k_spt, 1, 28, 28]
        x_spts = np.array(x_spts).astype(np.float32).reshape(batch_size, setsz, 1, resize, resize)
        y_spts = np.array(y_spts).astype(np.int64).reshape(batch_size, setsz)
        # [b, qrysz = n_way * k_query, 1, 28, 28]
        x_qrys = np.array(x_qrys).astype(np.float32).reshape(batch_size, querysz, 1, resize, resize)
        y_qrys = np.array(y_qrys).astype(np.int64).reshape(batch_size, querysz)
        #         print(x_qrys.shape)
        data_cache.append([x_spts, y_spts, x_qrys, y_qrys])

    return data_cache


def next(datasets, datasets_cache, indexes, mode='train'):
    # 如果所需的index超出当前已经获取的数量，则重新执行load_data_cache获取新的数据
    if indexes[mode] >= len(datasets_cache[mode]):
        indexes[mode] = 0
        datasets_cache[mode] = load_data_cache(datasets[mode], args)

    next_batch = datasets_cache[mode][indexes[mode]]
    indexes[mode] += 1

    return next_batch


# ------------------------------------------执行训练----------------------------------------
def main(args):
    # omniglot
    # 设置随机数种子
    random.seed(1337)
    np.random.seed(1337)
    # 加载训练集和测试集
    x_train = np.load('omniglot_train.npy')  # (973, 20, 1, 28, 28)
    x_val = np.load('omniglot_val.npy')  # (325, 20, 1, 28, 28)
    x_test = np.load('omniglot_test.npy')  # (325, 20, 1, 28, 28)
    datasets = {'train': x_train, 'val': x_val, 'test': x_test}

    datasets_cache = {"train": load_data_cache(x_train, args),  # current epoch data cached
                      "val": load_data_cache(x_val, args),
                      "test": load_data_cache(x_test, args)}

    # 全局参数设置
    n_way = args.n_way
    k_spt = args.k_spt  # support data 的个数
    task_num = args.task_num
    glob_update_step = args.glob_update_step
    glob_update_step_test = args.glob_update_step_test
    glob_meta_lr = args.glob_meta_lr  # 外循环学习率
    glob_base_lr = args.glob_base_lr  # 内循环学习率
    epochs = args.epochs

    indexes = {"train": 0, "val": 0, "test": 0}
    print("DB: train", x_train.shape, "validation", x_val.shape, "test", x_test.shape)
    # 开启0号GPU训练
    use_gpu = args.use_gpu
    paddle.set_device('gpu:0') if use_gpu else paddle.set_device('cpu')

    meta = MetaLearner(n_way, glob_update_step, glob_update_step_test, glob_meta_lr, glob_base_lr)
    best_acc = 0

    print('--------------------{}-way-{}-shot task start!---------------------'.format(n_way, k_spt))
    # for step in tqdm(range(epochs)):
    for step in range(epochs):
        # start = time.time()
        x_spt, y_spt, x_qry, y_qry = next(datasets, datasets_cache, indexes, 'train')
        x_spt = paddle.to_tensor(x_spt)
        y_spt = paddle.to_tensor(y_spt)
        x_qry = paddle.to_tensor(x_qry)
        y_qry = paddle.to_tensor(y_qry)
        accs, loss = meta(x_spt, y_spt, x_qry, y_qry)
        # end = time.time()
        if step % 100 == 0:
            print("epoch:", step)
            print(accs)
        #         print(loss)

        if step % 1000 == 0:
            accs = []
            for _ in range(1000 // task_num):
                x_spt, y_spt, x_qry, y_qry = next(datasets, datasets_cache, indexes, 'val')
                x_spt = paddle.to_tensor(x_spt)
                y_spt = paddle.to_tensor(y_spt)
                x_qry = paddle.to_tensor(x_qry)
                y_qry = paddle.to_tensor(y_qry)

                for x_spt_one, y_spt_one, x_qry_one, y_qry_one in zip(x_spt, y_spt, x_qry, y_qry):
                    test_acc = meta.finetunning(x_spt_one, y_spt_one, x_qry_one, y_qry_one)
                    accs.append(test_acc)

            print('---------------------在{}个随机任务上测试：---------------------'.format(np.array(accs).shape[0]))
            accs = np.array(accs).mean(axis=0).astype(np.float16)
            print('验证集准确率:', accs)
            print('------------------------------------------------------------')
            # 记录并保存最佳模型
            if accs[-1] > best_acc:
                best_acc = accs[-1]
                model_params = [item.numpy() for item in meta.net.vars]
                model_params_file = open('model/model_param_best_%sway%sshot.pkl' % (n_way, k_spt), 'wb')
                pickle.dump(model_params, model_params_file)
                model_params_file.close()
    print('The best acc on validation set is {}'.format(best_acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        usage='''you can set the global parameters to start the train program.''',
        description="this is the help of this script."
    )

    parser.add_argument("--n_way", type=int, default=5, help="The number of classes.")
    parser.add_argument("--k_spt", type=int, default=1, help="The number of shots.")
    parser.add_argument("--k_query", type=int, default=15, help="The number of query samples.")
    parser.add_argument("--task_num", type=int, default=32, help="The number of tasks to train together.")
    parser.add_argument("--glob_update_step", type=int, default=5, help="The global update step size.")
    parser.add_argument("--glob_update_step_test", type=int, default=5, help="The global update step size for test.")
    parser.add_argument("--glob_meta_lr", type=float, default=0.001, help="The global meta learning rate.")
    parser.add_argument("--glob_base_lr", type=float, default=0.1, help="The global base learning rate.")
    parser.add_argument("--epochs", type=int, default=10000, help="The number of training epochs.")
    parser.add_argument("--use_gpu", action="store_true", default='True', help="Do you want to use gpu.")
    args = parser.parse_args()

    main(args)
