import os
import sys
import argparse
import inspect
import datetime
import json
import numpy as np

import time

import paddle
import paddle.fluid as fluid
import paddle.fluid.optimizer as optimizer


#import models
import hmdb_2d_resnets


parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow')
parser.add_argument('-exp_name', type=str)
parser.add_argument('-video_dir', type=str, default='data/')
parser.add_argument('-file_dir', type=str, default='data/hmdb/')
parser.add_argument('-batch_size', type=int, default=24)
parser.add_argument('-length', type=int, default=16)
parser.add_argument('-learnable', type=str, default='[0,0,0,0,0]')
parser.add_argument('-niter', type=int)
parser.add_argument('-system', type=str)
parser.add_argument('-model', type=str)
parser.add_argument('-check_point', type=str)
parser.add_argument('-learning_rate', type = float, default = 0.01)
parser.add_argument('-movement', type = float, default = 0.9)

args = parser.parse_args()

##################
#
# Create model, dataset, and training setup
#
##################

data_path = args.video_dir       ##  path for train/test data.
file_path = args.video_dir       ##  path for train/test description file.

batch_size = args.batch_size

def concatdict(dictrep, dictbreprep):
    
    dictrep['repflow02._conv.weight'] = dictrep['repflow._conv.weight']
    dictrep['repflow02._unconv.weight'] = dictrep['repflow._unconv.weight']
    dictrep['repflow02.bottleneck.weight'] = dictrep['repflow.bottleneck.weight']
    dictrep['repflow02.unbottleneck.weight'] = dictrep['repflow.unbottleneck.weight']
    dictrep['repflow02.bn.weight'] = dictrep['repflow.bn.weight']
    dictrep['repflow02.bn.bias'] = dictrep['repflow.bn.bias']
    dictrep['repflow02.bn._mean'] = dictrep['repflow.bn._mean']
    dictrep['repflow02.bn._variance'] = dictrep['repflow.bn._variance']
    dictrep['repflow02.img_grad_conv2d.weight'] = dictrep['repflow.img_grad_conv2d.weight']
    dictrep['repflow02.img_grad2_conv2d.weight'] = dictrep['repflow.img_grad2_conv2d.weight']
    dictrep['repflow02.f_grad_conv2d.weight'] = dictrep['repflow.f_grad_conv2d.weight']
    dictrep['repflow02.f_grad2_conv2d.weight'] = dictrep['repflow.f_grad2_conv2d.weight']
    dictrep['repflow02.div_conv2d.weight'] = dictrep['repflow.div_conv2d.weight']
    dictrep['repflow02.div2_conv2d.weight'] = dictrep['repflow.div2_conv2d.weight']
    dictrep['repflow02.t_linear.weight'] = dictrep['repflow.t_linear.weight']
    dictrep['repflow02.l_linear.weight'] = dictrep['repflow.l_linear.weight']
    dictrep['repflow02.a_linear.weight'] = dictrep['repflow.a_linear.weight']
    
    return dictrep

place = fluid.CUDAPlace(0)
with fluid.dygraph.guard(place):

    def batch_generator_creator(dataset):
        dataset.shuffdata()
        def _batch_reader():
            for i in range(len(dataset)):
                data = dataset[i]
                if(data[0].shape[3] == 112):
                    yield data

        return (_batch_reader)

    if args.system == 'hmdb':
        from hmdb_lintel import HMDB as DS
        dataseta = DS(file_path, data_path, model=args.model, mode=args.mode, length=args.length)
        train_reader = paddle.batch(batch_generator_creator(dataseta),
                                batch_size=batch_size,
                                drop_last=False)
    
        dataset = DS(file_path, data_path, model=args.model, mode=args.mode, length=args.length, c2i=dataseta.class_to_id)
        test_reader  = paddle.batch(batch_generator_creator(dataset),
                                batch_size=batch_size,
                                drop_last=False)


    if args.system == 'minikinetics':
        train = '/home/aistudio/data/data/kinetics/minikinetics_train.json'
        val = '/home/aistudio/data/data/kinetics/minikinetics_val.json'
        root = '/home/aistudio/data/minikinetics/'
        from minikinetics_dataset import MK
        dataset_tr = MK(train, root, length=args.length, model=args.model, mode=args.mode)
        #dl = torch.utils.data.DataLoader(dataset_tr, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
        train_reader = paddle.batch(batch_generator_creator(dataset_tr),
                            batch_size=batch_size,
                            drop_last=True)
        dl = batch_generator_creator(dataset_tr)

        dataset = MK(val, root, length=args.length, model=args.model, mode=args.mode)
        #vdl = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
        vdl = batch_generator_creator(dataset)
        eval_reader  = paddle.batch(batch_generator_creator(dataset),
                            batch_size=batch_size,
                            drop_last=True)
        dataloader = {'train':dl, 'val':vdl}

    if args.system == 'kinetics':
        train = 'data/kinetics/kinetics_train.json'
        val = 'data/kinetics/kinetics_val.json'
        root = '/ssd/kinetics/'
        from minikinetics_dataset import MK
        dataset_tr = MK(train, root, length=args.length, model=args.model, mode=args.mode)
        #dl = torch.utils.data.DataLoader(dataset_tr, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
        # If the data generator yields a batch each time,
        # use DataLoader.set_batch_generator to set the data source.
        dl = batch_generator_creator(dataset_tr)

        dataset = MK(val, root, length=args.length, model=args.model, mode=args.mode)
        #vdl = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
        vdl = batch_generator_creator(dataset)
        dataloader = {'train':dl, 'val':vdl}


    repmodel = hmdb_2d_resnets.resnet50(pretrained=False, mode='rgb')  #
    
#exit()

    lr = args.learning_rate
    movement = args.movement
    lrdecay = fluid.dygraph.InverseTimeDecay(
                  learning_rate=lr,
                  decay_steps=3000,
                  decay_rate=0.5)
    #opt = optimizer.SGD(parameter_list=repmodel.parameters(), learning_rate=lr)
    opt = optimizer.MomentumOptimizer(parameter_list=repmodel.parameters(), 
              learning_rate=lr,
              momentum=movement)  #optim.SGD([{'params':params}, {'params':other, 'lr':0.01*lr}], lr=lr, weight_decay=1e-6, momentum=0.9)
#lr_sched = optim.lr_scheduler.ReduceLROnPlateau(solver, patience=7)


#################
#
# Setup logs, store model code
# hyper-parameters, etc...
#
#################
    log_name = datetime.datetime.today().strftime('%m-%d-%H%M%S')+'-'+args.exp_name
    log_path = os.path.join('logs/',log_name)
    os.mkdir(log_path)
    os.system('cp * logs/'+log_name+'/')

# deal with hyper-params...
    with open(os.path.join(log_path,'params.json'), 'w') as out:
        hyper = vars(args)
        json.dump(hyper, out)
    log = {'epoch acc':[], 'epoch loss':[], 'validation':[], 'train_acc':[]}
    

###############
#
# Train the model and save everything
#
###############
    num_epochs = 1
    c = 0
    phase = 'eval'
    

    if(args.check_point):
        check_point = os.path.join('logs/',args.check_point)
        #check_point = os.path.join(check_point,args.check_point)
        premodel, _ = fluid.dygraph.load_dygraph(os.path.join(check_point,'Myrepflow'))
        #repparms, _ = fluid.dygraph.load_dygraph('/home/aistudio/work/representation-flow/logs/08-26-140622-train/1000/Myrepflow')
        #newparms = concatdict(repparms, premodel)
        #print('[Final shot]', newparms['repflow02.t_linear.weight'])
        repmodel.set_dict(premodel)

    for epoch in range(num_epochs):

            tloss = 0.
            acc = 0.
            tot = 0
            step = 0
            tacc = 0
            e=s=0

            repmodel.eval()

            if phase=='eval':
                for batch_id,data in enumerate(test_reader()):    #for vid, cls in dataloader[phase]:
                    #vid = np.array([data[0]], np.float32)
                    vid = np.array([x[0] for x in data], np.float32)
                    cls = np.array([x[1] for x in data]).astype('int64')
                    cls = cls[:, np.newaxis]
                    vid = fluid.dygraph.to_variable(vid)
                    cls = fluid.dygraph.to_variable(cls)
                    cls.stop_gradient = True
                    
                    outputs = repmodel(vid)
                
                    loss = fluid.layers.cross_entropy(outputs, cls)
                    avg_loss = fluid.layers.mean(loss)
                    acc=fluid.layers.accuracy(outputs,cls, k=1)       #计算精度

                    tacc += acc.numpy()
                    tloss += avg_loss.numpy() #.item()
                   
                    c += 1
                    step += 1
                    #print('real', cls)
                    print('epoch',epoch,'step',c,'eval loss',avg_loss.numpy(), 'acc', acc.numpy())

                print('epoch',epoch,'epoch eval loss',tloss/step, 'epoch acc', tacc/step)
    



    #lr_sched.step()
