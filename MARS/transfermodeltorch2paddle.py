#需要在同时安装paddle和pytorch
import torch
import torchvision.models as models
import paddle
import paddle.fluid as fluid
from collections import OrderedDict
torch_weight = torch.load('RGB_Kinetics_16f.pth',map_location=torch.device('cpu'))#这里需要改成你下载的torch的权重的位置!!
for torch_key in torch_weight['state_dict'].keys():
    print(torch_key)
    
from models import resnext
with fluid.dygraph.guard():
    # 这里提供的参数示例是从Kinetics400预训练模型转化,下面三个分别是对RGB stream、Flow stream、MARS stream进行转化
    # paddle_model = resnext.resnet101(num_classes=400,shortcut_type='B',cardinality=32,
    #                sample_size=112,sample_duration=16,input_channels=3,output_layers=[],curr_mode='RGB'
    # paddle_model = resnext.resnet101(num_classes=400,shortcut_type='B',cardinality=32,
    #                sample_size=112,sample_duration=16,input_channels=2,output_layers=[],curr_mode='Flow'
    # paddle_model = resnext.resnet101(num_classes=400,shortcut_type='B',cardinality=32,
    #                sample_size=112,sample_duration=16,input_channels=3,output_layers=[],curr_mode='MARS'
    paddle_model = resnext.resnet101(parameter) # 这里需要对照训练过程传入的参数进行设置
    paddle_weight = paddle_model.state_dict()
    for paddle_key in paddle_weight:
        print(paddle_key)
    
    paddle_weight = paddle_model.state_dict()
    new_weight_dict = OrderedDict()
    for torch_key, paddle_key in zip(torch_weight['state_dict'].keys(), paddle_weight.keys()):
        if torch_key.find('fc') > -1:
            # paddle的fc层的weight与竞品不太一致，需要转置一下
            new_weight_dict[paddle_key] =    torch_weight['state_dict'][torch_key].detach().numpy().T
        else:
            new_weight_dict[paddle_key] = torch_weight['state_dict'][torch_key].detach().numpy()
    paddle_model.set_dict(new_weight_dict)
    fluid.dygraph.save_dygraph(paddle_model.state_dict(),"RGB_Kinetics_16f")#修改成自己的预训练模型
print('OK!!!')