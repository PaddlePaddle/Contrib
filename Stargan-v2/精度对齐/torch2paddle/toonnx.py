#coding: utf-8
import torch
#import torchvision
# 1.导入pytorch模型定义
from nasnet_mobile import nasnetamobile 
# 2.指定输入大小的shape
dummy_input = torch.randn(1, 3, 256, 256) 

# 3. 构建pytorch model
model = nasnetamobile(121,pretrained=False)
# 4. 载入模型参数
model.load_state_dict(torch.load('/home/aistudio/data/data23875/nasnet_mobile.pkl', map_location='cpu'))

# 5.导出onnx模型文件
torch.onnx.export(model, dummy_input, "nasnet.onnx",verbose=True)
