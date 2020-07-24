#导入torch的包
import torch
import torchvision.models as models
#导入paddle的包
import paddle
import paddle.fluid as fluid
from collections import OrderedDict
# 加载torch权重
#这里需要改成你下载的torch的权重的位置!!
torch_weight = torch.load('RGB_Kinetics_16f.pth',map_location=torch.device('cpu'))
#将torch权重的参数列表打印出来
for torch_key in torch_weight['state_dict'].keys():
    print(torch_key)
    
from models.model import generate_model
with fluid.dygraph.guard():
    #加载paddle网络结构
    paddle_model = generate_model(参数)#这里需要对照训练过程传入的参数进行设置
    #读取paddle网络结构的参数列表
    paddle_weight = paddle_model.state_dict()
    #将paddle权重的参数列表打印出来
    for paddle_key in paddle_weight:
        print(paddle_key)
    
    #读取paddle网络结构的参数列表
    paddle_weight = paddle_model.state_dict()
    #进行模型参数转换
    new_weight_dict = OrderedDict()
    for torch_key, paddle_key in zip(torch_weight['state_dict'].keys(), paddle_weight.keys()):
        if torch_key.find('fc') > -1:
            # paddle的fc层的weight与竞品不太一致，需要转置一下
            new_weight_dict[paddle_key] =    torch_weight['state_dict'][torch_key].detach().numpy().T
        else:
            new_weight_dict[paddle_key] = torch_weight['state_dict'][torch_key].detach().numpy()
    paddle_model.set_dict(new_weight_dict)
    fluid.dygraph.save_dygraph(paddle_model.state_dict(),"RGB_Kinetics_16f")#修改成自己的路径
print('OK!!!')