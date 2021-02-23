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
parser.add_argument('-data_dir', type=str, default='data/data/')
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


##################
#
# Create model, dataset, and training setup
#
##################

data_path = args.data_dir       ##  path for train/test data.

batch_size = args.batch_size

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
        dataseta = DS('data/hmdb/split_train.txt', data_path, model=args.model, mode=args.mode, length=args.length)
        train_reader = paddle.batch(batch_generator_creator(dataseta),
                                batch_size=batch_size,
                                drop_last=False)
    
        dataset = DS('data/hmdb/split_test.txt', data_path, model=args.model, mode=args.mode, length=args.length, c2i=dataseta.class_to_id)
        eval_reader  = paddle.batch(batch_generator_creator(dataset),
                                batch_size=batch_size,
                                drop_last=False)


    repmodel = hmdb_2d_resnets.resnet50(pretrained=False, mode='rgb')  

    lr = args.learning_rate
    momentum = args.momentum
    lrdecay = fluid.dygraph.InverseTimeDecay(
                  learning_rate=lr,
                  decay_steps=3000,
                  decay_rate=0.5)
    
    opt = optimizer.MomentumOptimizer(parameter_list=repmodel.parameters(), 
              learning_rate=lr,
              momentum=momentum)  


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
        premodel, _ = fluid.dygraph.load_dygraph(os.path.join(check_point,'Myrepflow'))
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
 
