from .TPN import TPN
from .backbone import ResNet_SlowFast
from paddle import fluid
from paddle.fluid.param_attr import ParamAttr
from collections import OrderedDict

import numpy as np
import math
r50f8s8={
    'TPN':{'out_channels':1024},
    'temporal_modulation_config':{'down_scale':[8,8]},
    'level_fusion':{'mid_channels':1024,'out_channels':2048},
    'aux_head':{'planes':400,'loss_weight':0.5,'drop_ratio':0.5},
    'cls_head':{'planes':400,'drop_ratio':0.5},
    'sample':{'seglen':8,'step':8}
}
r50f32s2={
    'TPN':{'out_channels':1024},
    'temporal_modulation_config':{'down_scale':[32,32]},
    'level_fusion':{'mid_channels':1024,'out_channels':2048},
    'aux_head':{'planes':400,'loss_weight':0.5,'drop_ratio':0.5},
    'cls_head':{'planes':400,'drop_ratio':0.5},
    'sample':{'seglen':32,'step':2}
}


class TSN3D():
    def __init__(self, config=None,layers=50, is_training=True):
        self.layers = layers
       
        self.is_training = is_training
        self.config = config
    def Cls_head(self,input,label):
        classdim = self.config['cls_head']['planes']
        drop_ratio = self.config['cls_head']['drop_ratio']

    #    input = fluid.layers.reshape(input, (input.shape[0], -1))
        input = fluid.layers.flatten(input)
        input = fluid.layers.dropout(input, drop_ratio, is_test=(not self.is_training),
                                     dropout_implementation='upscale_in_train')
        fc_in_channel = input.shape[1]
        stdv = 1.0 / math.sqrt(fc_in_channel * 1.0)
        out = fluid.layers.fc(input,classdim,bias_attr=ParamAttr('cls_head_fc_b'),param_attr=ParamAttr(name='cls_head_fc_w',
                                                    initializer=fluid.initializer.Uniform(-stdv, stdv)))
        loss = fluid.layers.softmax_with_cross_entropy(out,label)
        loss = fluid.layers.reduce_mean(loss,dim=0)
        return out,loss

    def net(self,data,label,data_format='NCDHW'):
        seg_num = data.shape[1]
        beckbone = ResNet_SlowFast(self.layers,self.is_training)
        neck = TPN(self.config,seg_num,self.is_training)
        data = beckbone.net(data)
        data,loss_aux= neck.net(data,label)

        #将输出进行全局平均池化
        data = fluid.layers.pool3d(input=data,pool_type='avg',pool_size=(1,7,7),global_pooling=True)

        #将seg_num个输出进行平均融合
        data = fluid.layers.reshape(x=data,shape=[-1, seg_num,data.shape[-4],data.shape[-3],data.shape[-2],data.shape[-1]])
        data = fluid.layers.reduce_mean(input=data,dim=1)

        #进行最终的分类
        data,loss_cls = self.Cls_head(data,label)
        acc = fluid.layers.accuracy(data,label)

        return data,acc,loss_aux,loss_cls

def get_pretrain_para(path='../pretrain/tpn.pkl'):
    model = torch.load(path)
    state_dict = model.state_dict()
    f = open('para_pretrain.txt', 'w')
    for key in state_dict.keys():
        if not('tracked' in  key):
            f.write(key + '\n')
    f.close
    sub_key = []
    # 取出backbone的参数
    for key in state_dict.keys():
        if 'backbone' in key:
            sub_key.append(key)
    # 查找conv2的位置
    conv2_loc = []
    for i in range(len(sub_key)):
        if 'conv2' in sub_key[i]:
            conv2_loc.append(i)
    # 将conv2的位置往后移动5个
    for i in range(len(conv2_loc)):
        index = conv2_loc[i]
        item = sub_key[index]
        sub_key.pop(index)
        sub_key.insert(index + 5, item)
    # 构建新的backbone字典
    backbone_para = OrderedDict()
    for i in range(len(sub_key)):
        if not ('tracked' in sub_key[i]):
            backbone_para[sub_key[i]] = state_dict[sub_key[i]].numpy()
        del state_dict[sub_key[i]]

    #构建TPN字典
    #Spatialmodule 字段
    TPN_para = OrderedDict()
    for key in state_dict.keys():
        if 'spatial_modulation' in key:
            if not('tracked' in key):
                TPN_para[key] = state_dict[key].numpy()

    #构建Temopralmodule字段
    for key in state_dict.keys():
        if 'temporal_modulation' in key:
            if not ('tracked' in key):
                TPN_para[key] = state_dict[key].numpy()

    #构建top_down字段
    for key in state_dict.keys():
        if 'level_fusion_op2' in key:
            if not ('tracked' in key):
                TPN_para[key] = state_dict[key].numpy()

    #构建downsamping字段
    for key in state_dict.keys():
        if 'downsampling_ops' in key:
            if not ('tracked' in key):
                TPN_para[key] = state_dict[key].numpy()

    #构建down_top字段
    for key in state_dict.keys():
        if 'level_fusion_op' in key:
            if not ('tracked' in key):
                TPN_para[key] = state_dict[key].numpy()

    #构建pyramid_fusion字段
    for key in state_dict.keys():
        if 'pyramid_fusion_op' in key:
            if not ('tracked' in key):
                TPN_para[key] = state_dict[key].numpy()

    #构建aux_head字段
    for key in state_dict.keys():
        if 'aux_head' in key:
            if not ('tracked' in key):
                if  ('fc.weight' in key):
                    TPN_para[key] = state_dict[key].numpy().T
                else:
                    TPN_para[key] = state_dict[key].numpy()

    #构建cls_head字段
    for key in state_dict.keys():
        if 'cls_head' in key:
            if ('weight' in key):
                TPN_para[key] = state_dict[key].numpy().T
            else:
                TPN_para[key] = state_dict[key].numpy()


    para_dict = OrderedDict()
    para_dict.update(backbone_para)
    para_dict.update(TPN_para)


    f = open('para_pretrain', 'w')
    for key in para_dict.keys():
        f.write(key + '\n')
    f.close
    return para_dict,model

if __name__ == "__main__":
    startup = fluid.Program()
    train_prog = fluid.Program()
    with fluid.program_guard(train_prog, startup):
        with fluid.unique_name.guard():
            input = fluid.data(name='image', shape=(2,1,24,224,224), dtype='float32')
            label = fluid.data(name='label', shape=(2,1), dtype='int64')
            model = TSN3D(config=r50f8s8,is_training=False)
            fetch_list = model.net(input,label)

    place = fluid.CUDAPlace(0)
  #  place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup)
    para_list = train_prog.all_parameters()
    f = open('para_p.txt','w')
    for i in range(len(para_list)):
        f.write(para_list[i].name+'\n')
    f.close()


    pretrain_para,model = get_pretrain_para()
    para_list = train_prog.all_parameters()
    pretrain_para_new = OrderedDict()
    i = 0
    for key in pretrain_para:
        pretrain_para_new[para_list[i].name] = pretrain_para[key]
        i = i+1
    fluid.io.set_program_state(train_prog,pretrain_para_new)
    data = np.random.randn(2,1,24,224,224).astype('float32')
#    data = np.ones(shape=(2,1,24,224,224),dtype='float32')
    label = np.array([[10],[20]]).astype('int64')
    outs,aux_loss,cls_loss = exe.run(train_prog, fetch_list=fetch_list,feed={'image':data,'label':label})
    print(outs.shape)
    #
    x = torch.from_numpy(data)
    l = torch.from_numpy(label)
    x = torch.reshape(x,(2,1,8,3,224,224))
    x = torch.Tensor.permute(x,(0,1,3,2,4,5))

    model.eval()
    loss ,cls_score= model([1], True, img_group_0=x, gt_label=l)
    cls_score = torch.detach(cls_score).numpy()
    print(loss)
    print(aux_loss,cls_loss)
    diff = np.absolute(cls_score-outs)
    mean = np.mean(diff)
    print(mean)




    fluid.io.save_inference_model(dirname='./', feeded_var_names=['image','label'],
                                  target_vars=fetch_list,
                                  program_only=True, executor=exe, main_program=train_prog)



