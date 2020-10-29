import torch
import paddle.fluid as fluid
from collections import OrderedDict

from networks import *

torch_weight = torch.load('kinetics400_tpn_r50f32s2.pth', map_location=torch.device('cpu'))
print(torch_weight.keys())
print(len(torch_weight.keys()))

#with fluid.dygraph.guard():
    # generator = Generator(w_hpf=0)
    # mapping_network = MappingNetwork(num_domains=3)
    # style_encoder = StyleEncoder(num_domains=3)
    #
    # generator_weight = generator.state_dict()
    # mapping_network_weight = mapping_network.state_dict()
    # style_encoder_weight = style_encoder.state_dict()

    # print('===============generator_weight================')
    # print(len(generator_weight))
    # # for paddle_key in generator_weight:
    # # print(paddle_key)
    #
    # print('===============mapping_network_weight================')
    # print(len(mapping_network_weight))
    # # for paddle_key in mapping_network_weight:
    # # print(paddle_key)
    # print('===============style_encoder_weight================')
    # print(len(style_encoder_weight))
    # # for paddle_key in style_encoder_weight:
    # # print(paddle_key)
    #
    # new_generator_weight_dict = OrderedDict()
    # new_mapping_network_weight = OrderedDict()
    # new_style_encoder_weight = OrderedDict()
    #
    # for torch_key, paddle_key in zip(torch_weight['generator'].keys(), generator_weight.keys()):
    #     if torch_key.find('norm') and torch_key.find('weight') > 0:
    #         torch_key.replace('weight', 'scale')
    #
    #     if torch_key.find('fc') > 0:
    #         new_generator_weight_dict[paddle_key] = torch_weight['generator'][torch_key].detach().numpy().T
    #     else:
    #         new_generator_weight_dict[paddle_key] = torch_weight['generator'][torch_key].detach().numpy()
    #     #print(paddle_key, '=======', torch_key)
    #
    # for torch_key, paddle_key in zip(torch_weight['mapping_network'].keys(), mapping_network_weight.keys()):
    #     if torch_key.find('weight') > 0:
    #         new_mapping_network_weight[paddle_key] = torch_weight['mapping_network'][torch_key].detach().numpy().T
    #     else:
    #         new_mapping_network_weight[paddle_key] = torch_weight['mapping_network'][torch_key].detach().numpy()
    #
    # for torch_key, paddle_key in zip(torch_weight['style_encoder'].keys(), style_encoder_weight.keys()):
    #     if torch_key.find('unshared') >= 0:
    #         new_style_encoder_weight[paddle_key] = torch_weight['style_encoder'][torch_key].detach().numpy().T
    #     else:
    #         new_style_encoder_weight[paddle_key] = torch_weight['style_encoder'][torch_key].detach().numpy()
    #     print(paddle_key, '=======', torch_key)
    # generator.set_dict(new_generator_weight_dict)
    # fluid.dygraph.save_dygraph(generator.state_dict(), './generator_torch')
    #
    # mapping_network.set_dict(new_mapping_network_weight)
    # fluid.dygraph.save_dygraph(mapping_network.state_dict(), './mapping_network_torch')
    #
    # style_encoder.set_dict(new_style_encoder_weight)
    # fluid.dygraph.save_dygraph(style_encoder.state_dict(), './style_encoder_torch')

    ######################FAN########################
    # fan = FAN()
    #
    # fan_weight = fan.state_dict()
    # print(len(fan_weight))
    # # for paddle_key in fan_weight:
    # #     print(paddle_key, ':', fan_weight[paddle_key].shape)
    #
    # new_fan_weight_dict = OrderedDict()
    #
    # # for parameters in torch_weight['fan']:
    # #     print(parameters, ":", torch_weight['fan'][parameters].size())
    #
    # for torch_key in torch_weight['fan'].keys():
    #     for paddle_key in fan_weight.keys():
    #
    #         if torch_key.find('running_mean') and paddle_key.find('_mean')  > 0:
    #             tmp_key = torch_key.replace('running_mean','_mean')
    #             if tmp_key==paddle_key:
    #                 new_fan_weight_dict[paddle_key] = torch_weight['fan'][torch_key].detach().numpy()
    #                 print('_mean:', torch_key, ':::::::', paddle_key)
    #
    #         if torch_key.find('running_var') and paddle_key.find('_variance') > 0:
    #             tmp_key = torch_key.replace('running_var', '_variance')
    #             if tmp_key == paddle_key:
    #                 new_fan_weight_dict[paddle_key] = torch_weight['fan'][torch_key].detach().numpy()
    #                 print('_variance:', torch_key, ':::::::', paddle_key)
    #         if torch_key == paddle_key:
    #             new_fan_weight_dict[paddle_key] = torch_weight['fan'][torch_key].detach().numpy()
    #         # if torch_key.find('fc') > 0:
    #         #     new_fan_weight_dict[paddle_key] = torch_weight['fan'][torch_key].detach().numpy().T
    #         # else:
    #         #     new_fan_weight_dict[paddle_key] = torch_weight['fan'][torch_key].detach().numpy()
    #
    # fan.set_dict(new_fan_weight_dict)
    # fluid.dygraph.save_dygraph(fan.state_dict(), './fan')
