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
import flow_2d_resnets
import hmdb_2d_model
import hmdb_2d_resnets


parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow')
parser.add_argument('-exp_name', type=str)
parser.add_argument('-batch_size', type=int, default=24)
parser.add_argument('-length', type=int, default=16)
parser.add_argument('-learnable', type=str, default='[0,0,0,0,0]')
parser.add_argument('-niter', type=int)
parser.add_argument('-system', type=str)
parser.add_argument('-model', type=str)
parser.add_argument('-check_point', type=str)
parser.add_argument('-learning_rate', type = float, default = 0.01)
parser.add_argument('-momentum', type = float, default = 0.9)

args = parser.parse_args()



#place = fluid.CUDAPlace()

##################
#
# Create model, dataset, and training setup
#
##################

batch_size = args.batch_size

os.environ['ffmpegpath'] = '/home/aistudio/work/FFmpeg'
os.environ['PATH'] = os.environ['PATH'] + ':' + os.environ['ffmpegpath'] +'/bin'
os.environ['LD_LIBRARY_PATH'] = os.environ['LD_LIBRARY_PATH'] + ':' + os.environ['ffmpegpath'] +'/lib'
os.environ['CPATH'] = os.environ['ffmpegpath'] +'/include'
os.environ['LIBRARY_PATH'] = os.environ['LIBRARY_PATH'] + ':' + os.environ['ffmpegpath'] +'/lib'

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
        dataseta = DS('/home/aistudio/work/representation-flow/data/hmdb/split1_train.txt', '/home/aistudio/data/data49479/', model=args.model, mode=args.mode, length=args.length)
        train_reader = paddle.batch(batch_generator_creator(dataseta),
                                batch_size=batch_size,
                                drop_last=False)
    
        dataset = DS('/home/aistudio/work/representation-flow/data/hmdb/split1_test.txt', '/home/aistudio/data/data49479/', model=args.model, mode=args.mode, length=args.length, c2i=dataseta.class_to_id)
        eval_reader  = paddle.batch(batch_generator_creator(dataset),
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


    repmodel = hmdb_2d_resnets.resnet50(pretrained=False, mode='rgb')  
    #repmodel = hmdb_2d_model.ResNet(name_scope='resnet', layers=50, class_dim=51)   #.resnet_3d_v1(50, 51)  #flow_2p1d_resnets.resnet50(pretrained=False, mode=args.mode, n_iter=args.niter, learnable=eval(args.learnable), num_classes=400)
    # scale lr for flow layer
    params = repmodel.parameters()
    params = [p for p in params]
    other = []
    print('Params :', len(params))
    #ln = eval(args.learnable)
    
#exit()

    lr = args.learning_rate
    momentum = args.momentum
    lrdecay = fluid.dygraph.InverseTimeDecay(
                  learning_rate=lr,
                  decay_steps=3000,
                  decay_rate=0.5)
    #opt = optimizer.SGD(parameter_list=repmodel.parameters(), learning_rate=lr)
    opt = optimizer.MomentumOptimizer(parameter_list=repmodel.parameters(), 
              learning_rate=lr,
              momentum=momentum)  #optim.SGD([{'params':params}, {'params':other, 'lr':0.01*lr}], lr=lr, weight_decay=1e-6, momentum=0.9)
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
    log = {'epoch acc':[], 'epoch loss':[], 'val loss':[], 'val acc':[]}
    

###############
#
# Train the model and save everything
#
###############
    num_epochs = 20
    c = 0
    #
    

    if(args.check_point):
        check_point = os.path.join('logs/',args.check_point)
        #check_point = os.path.join(check_point,args.check_point)
        premodel, _ = fluid.dygraph.load_dygraph(os.path.join(check_point,'Myrepflow'))
        #repparms, _ = fluid.dygraph.load_dygraph('/home/aistudio/work/representation-flow/logs/08-26-140622-train/1000/Myrepflow')
        #newparms = concatdict(repparms, premodel)
        #print('[Final shot]', newparms['repflow02.t_linear.weight'])
        repmodel.set_dict(premodel)

    for epoch in range(num_epochs):
            phase = 'train'
            if phase=='train':
                repmodel.train()
                tloss = 0.
                acc = 0.
                tot = 0
                step = 0
                tacc = 0
                e=s=0
                for batch_id,data in enumerate(train_reader()):    #for vid, cls in dataloader[phase]:
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
                    avg_loss.backward()
                    for name, parms in repmodel.named_parameters():
                        if(name == 'repofrepflow.rep_flow.t_linear.weight'):
                            print(name, parms.numpy())
                    
                    opt.minimize(avg_loss)
                    repmodel.clear_gradients()
                    c += 1
                    step += 1
                    print('epoch',epoch,'step',step,'train loss',avg_loss.numpy(), 'acc', acc.numpy())
                
                log['epoch loss'].append((tloss/(step+1e-12)).tolist())
                log['epoch acc'].append((tacc/(step+1e-12)).tolist())

                param_path = os.path.join(log_path, str(epoch))
                fluid.dygraph.save_dygraph(repmodel.state_dict(),os.path.join(param_path,'Myrepflow'))#保存模型
                
                with open(os.path.join(log_path,'log.json'), 'w') as out:
                    json.dump(log, out)
                print('epoch',epoch,'epoch train loss',tloss/(step + 1e-12), 'epoch acc', tacc/(step + 1e-12))

    with open(os.path.join(log_path,'log.json'), 'w') as out:
        json.dump(log, out)
    #lr_sched.step()
