#copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import argparse
from FID.compute_fid import *
from collections import OrderedDict
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_data_path1',
                        type=str,
                        default='/home/j/Desktop/stargan-v2-master/data/celeba_hq/train/female',
                        help='path of image data')
    parser.add_argument('--image_data_path2',
                        type=str,
                        default='/home/j/Desktop/stargan-v2-master/expr/eval/celeba_hq/male2female',
                        help='path of image data')
    parser.add_argument('--inference_model',
                        type=str,
                        default='./pretrained/InceptionV3.pdparams',
                        help='path of inference_model.')
    parser.add_argument('--use_gpu',
                        type=bool,
                        default=True,
                        help='default use gpu.')
    parser.add_argument('--batch_size',
                        type=int,
                        default=100,
                        help='sample number in a batch for inference.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    path1 = args.image_data_path1
    path2 = args.image_data_path2
    paths = (path1, path2)
    inference_model_path = args.inference_model
    batch_size = args.batch_size

    with fluid.dygraph.guard():
        fid_value = calculate_fid_given_paths(paths, inference_model_path,
                                              batch_size, args.use_gpu, 2048)
        print('FID: ', fid_value)


if __name__ == "__main__":
    main()
    # args = parse_args()
    # path1 = args.image_data_path1
    # path2 = args.image_data_path2
    # inference_model_path = args.inference_model
    # batch_size = args.batch_size
    # print('Calculating FID for all tasks...')
    # fid_values = OrderedDict()
    # domains = os.listdir('/home/j/Desktop/stargan-v2-master/data/celeba_hq/val')#args.val_img_dir
    # domains.sort()
    # num_domains = len(domains)
    # for trg_domain in domains:
    #     src_domains = [x for x in domains if x != trg_domain]
    #
    #     for src_domain in src_domains:
    #         task = '%s2%s' % (src_domain, trg_domain)
    #         path_real = os.path.join('/dataset/afhq/train', trg_domain)#args.train_img_dir
    #         path_fake = os.path.join('/result/eval', task) #args.eval_dir
    #         print('Calculating FID for %s...' % task)
    #
    #         fid_value = calculate_fid_given_paths([path_real, path_fake], inference_model_path,
    #                                               batch_size, args.use_gpu, 2048)
    #         fid_values['FID_%s/%s' % ('latent', task)] = fid_value
    #
    # # calculate the average FID for all tasks
    # fid_mean = 0
    # for _, value in fid_values.items():
    #     fid_mean += value / len(fid_values)
    # fid_values['FID_%s/mean' % 'mode'] = fid_mean
    #
    # # report FID values
    # filename = os.path.join(args.eval_dir, 'FID_%.5i_%s.json' % ('f', 'latent'))
    # with open(filename, 'w') as f:
    #     json.dump(filename, f, indent=4, sort_keys=False)
