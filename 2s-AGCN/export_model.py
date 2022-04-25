#encoding=utf8
# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import argparse
import paddle

from paddle_model.agcn import Model

import warnings
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description='Model export.')
    parser.add_argument("--graph", type=str, default='graph.ntu_rgb_d.Graph')
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='The directory for saving the exported model',
        type=str,
        default='./output')

    parser.add_argument(
        '--model_path',
        dest='model_path',
        help='The path of model for export',
        type=str,
        default=None)

    parser.add_argument(
        '--batch',
        dest='batch',
        help='The infer batch',
        type=int,
        default=10)

    parser.add_argument('--pretrained', default=".")  #
    parser.add_argument('--save-inference-dir', default=".")  #
    return parser.parse_args()


def main(args):

    # build model
    model = Model(graph=args.graph)

    if args.model_path is not None:
        model.load_dict(paddle.load(args.model_path))
        print('Loaded trained params of model successfully.')
    shape = [args.batch, 3, 300, 25, 2]
    new_net = model
    new_net.eval()
    new_net = paddle.jit.to_static(
        new_net,
        input_spec=[paddle.static.InputSpec(shape=shape, dtype='float32')])
    save_path = os.path.join(args.save_dir, 'model')
    paddle.jit.save(new_net, save_path)
    print(f'Model is saved in {args.save_dir}.')


if __name__ == '__main__':
    args = parse_args()
    main(args)
