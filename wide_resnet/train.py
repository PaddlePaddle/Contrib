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

import paddle
from paddle.vision import transforms
from wide_resnet import WideResNet
import time
import os
import numpy as np
import paddle.distributed as dist

dist.get_world_size()
dist.init_parallel_env()

CUDA = True
if CUDA:
    paddle.set_device('gpu')
place = paddle.CUDAPlace(0) if CUDA else paddle.CPUPlace()

mean,std = ([0.4914, 0.4822, 0.4465],[0.2023, 0.1994, 0.2010])
mean = list(map(lambda x:x*255,mean))
std = list(map(lambda x:x*255,std))

train_loader = paddle.io.DataLoader(
    paddle.vision.datasets.Cifar10(mode='train', transform=transforms.Compose([
        # transforms.Resize(size=64),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.Transpose(order=(2,0,1)),
        transforms.Normalize(mean=mean,std=std),
    ]), download=True),
    places=place,batch_size=128, shuffle=True,
    num_workers=0, use_shared_memory=True)

val_loader = paddle.io.DataLoader(
    paddle.vision.datasets.Cifar10(mode='test', transform=transforms.Compose([
        transforms.Transpose(order=(2,0,1)),
        transforms.Normalize(mean=mean,std=std),
    ])), places=place,
    batch_size=128, shuffle=False,
    num_workers=0, use_shared_memory=True)

def training():
    global save_every
    model.train()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    best_acc=0
    losses = []
    for epoch in range(epochs):
        if epoch > 180:
            save_every = 2
        # train for one epoch
        start = time.time()
        # model.train()
        for _iter,(x,y) in enumerate(train_loader):
            y = paddle.reshape(y, (-1, 1))
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            opt.clear_grad()
            losses.append(np.mean(loss.numpy()))
            if _iter % 10 == 0:
                print('iter:%d  loss:%.4f'%(_iter,np.mean(losses)))
        print('Time per epoch:%.2f,loss:%.4f'%(time.time()-start,np.mean(losses)))

        if (epoch+1)%save_every==0 or epoch+1==epochs:
            # evaluate on validation set
            eval_acc,eval_loss = test()
            print("Validation accuracy/loss: %.2f%%,%.4f"%(eval_acc, eval_loss))
            model.train()
            paddle.save(model.state_dict(),os.path.join(save_dir, 'checkpoint_{}.pdparams'.format(epoch)))
            paddle.save(opt.state_dict(),os.path.join(save_dir, 'checkpoint_{}.pdopt'.format(epoch)))
            if eval_acc > best_acc:
                paddle.save(model.state_dict(),os.path.join(save_dir, 'checkpoint.pdparams'))
                paddle.save(opt.state_dict(),os.path.join(save_dir, 'checkpoint.pdopt')) 
            best_acc = max(eval_acc, best_acc)      
        scheduler.step()
    paddle.save(model.state_dict(),os.path.join(save_dir, 'model.pdparams'))
    paddle.save(opt.state_dict(),os.path.join(save_dir, 'model.pdopt'))         
    print('Best accuracy on validation dataset: %.2f%%'%(best_acc))

def test():
    model.eval()
    accuracies = []
    losses = []
    for (x,y) in val_loader:
        with paddle.no_grad():
            logits = model(x)
            y = paddle.reshape(y, (-1, 1))
            loss = loss_fn(logits, y)
            acc = acc_fn(logits, y)
            accuracies.append(np.mean(acc.numpy()))
            losses.append(np.mean(loss.numpy()))
    return np.mean(accuracies)*100, np.mean(losses) 

import warnings

warnings.filterwarnings("ignore", category=Warning)
model = WideResNet(28,10,20,0.3)
model = paddle.DataParallel(model)

epochs = 400
save_every = 5
loss_fn = paddle.nn.CrossEntropyLoss()
acc_fn = paddle.metric.accuracy
scheduler=paddle.optimizer.lr.PiecewiseDecay(boundaries=[60,120,160,200,240,260,280],values=[0.05,0.01,0.002,0.0004,0.0002,0.0001,0.00005],verbose=True)
opt = paddle.optimizer.Momentum(parameters=model.parameters(), learning_rate=scheduler, momentum=0.9,weight_decay=0.0005)
save_dir = '/home/aistudio/models/cifar10/ResNet_wide'

training()
