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
import pickle
import argparse
import numpy as np
import auto_log


from paddle import inference
from paddle.inference import Config, create_predictor

import warnings
warnings.filterwarnings("ignore")


def parse_args():
    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    # general params
    parser = argparse.ArgumentParser("2sAGCN inference model script")
    parser.add_argument('-c',
                        '--config',
                        type=str,
                        default='configs/example.yaml',
                        help='config file path')
    parser.add_argument("--data_file", default='./data/ntu/tiny_dataset/tiny_infer_data.npy', type=str, help="input data path")
    parser.add_argument("--label_file", default='./data/ntu/tiny_dataset/tiny_infer_label.pkl', type=str, help="input label path")
    parser.add_argument("--model_file", default='./output/model.pdmodel', type=str)
    parser.add_argument("--params_file", default='./output/model.pdiparams', type=str)

    # params for predict
    parser.add_argument("-b", "--batch-size", type=int, default=10)
    parser.add_argument("--use-gpu", type=str2bool, default=False)
    parser.add_argument("--precision", type=str, default="fp32")
    parser.add_argument("--ir_optim", type=str2bool, default=True)
    parser.add_argument("--use_tensorrt", type=str2bool, default=False)
    parser.add_argument("--gpu_mem", type=int, default=8000)
    parser.add_argument("--benchmark", type=str2bool, default=True)
    parser.add_argument("--enable_mkldnn", type=str2bool, default=False)
    parser.add_argument("--cpu_threads", type=int, default=None)

    parser.add_argument("--model-dir", default='./output/model.pdiparams', type=str)    #
    return parser.parse_args()


def create_paddle_predictor(args):
    config = Config(args.model_file, args.params_file)
    if args.use_gpu:
        config.enable_use_gpu(args.gpu_mem, 0)
    else:
        config.disable_gpu()
        if args.cpu_threads:
            config.set_cpu_math_library_num_threads(args.cpu_threads)
        if args.enable_mkldnn:
            # cache 10 different shapes for mkldnn to avoid memory leak
            config.set_mkldnn_cache_capacity(10)
            config.enable_mkldnn()
            if args.precision == "fp16":
                config.enable_mkldnn_bfloat16()

    # config.disable_glog_info()
    config.switch_ir_optim(args.ir_optim)  # default true
    if args.use_tensorrt:
        # choose precision
        if args.precision == "fp16":
            precision = inference.PrecisionType.Half
        elif args.precision == "int8":
            precision = inference.PrecisionType.Int8
        else:
            precision = inference.PrecisionType.Float32

        # calculate real max batch size during inference when tenrotRT enabled
        num_seg = 1
        num_views = 1
        max_batch_size = args.batch_size * num_views * num_seg
        config.enable_tensorrt_engine(precision_mode=precision,
                                      max_batch_size=max_batch_size)

    config.enable_memory_optim()
    # use zero copy
    config.switch_use_feed_fetch_ops(False)
    predictor = create_predictor(config)

    return config, predictor


def parse_file_paths(data_path, label_path, use_mmap=True):
    try:
        with open(label_path) as f:
            sample_name, label = pickle.load(f)
    except:
        # for pickle file from python2
        with open(label_path, 'rb') as f:
            sample_name, label = pickle.load(f, encoding='latin1')

    # load data
    if use_mmap:
        data = np.load(data_path, mmap_mode='r')
    else:
        data = np.load(data_path)
    return data, sample_name, label


def main():
    args = parse_args()

    model_name = '2sAGCN'
    print(f"Inference model({model_name})...")
    # InferenceHelper = build_inference_helper(cfg.INFERENCE)

    inference_config, predictor = create_paddle_predictor(args)

    # get data
    data, sample_name, label = parse_file_paths(data_path=args.data_file, label_path=args.label_file)
    data = data[-100:]
    sample_name = sample_name[-100:]
    label = label[-100:]
    # ps:这里没对大小做检查，希望它大于100

    if args.benchmark:
        num_warmup = 0

        # instantiate auto log
        pid = os.getpid()
        autolog = auto_log.AutoLogger(
            model_name="2sAGCN",
            model_precision=args.precision,
            batch_size=args.batch_size,
            data_shape="dynamic",
            save_path="./output/auto_log.lpg",
            inference_config=inference_config,
            pids=pid,
            process_name=None,
            gpu_ids=0 if args.use_gpu else None,
            time_keys=['preprocess_time', 'inference_time', 'postprocess_time'],
            warmup=num_warmup)

    # Inferencing process
    batch_num = args.batch_size
    acc = []
    for st_idx in range(0, data.shape[0], batch_num):
        ed_idx = min(st_idx + batch_num, data.shape[0])

        # auto log start
        if args.benchmark:
            autolog.times.start()

        # Pre process batched input
        batched_inputs = [data[st_idx:ed_idx]]
        batch_label = label[st_idx:ed_idx]
        batch_sample_name = sample_name[st_idx:ed_idx]

        if args.benchmark:
            autolog.times.stamp()

        # run inference
        input_names = predictor.get_input_names()
        for i, name in enumerate(input_names):
            input_tensor = predictor.get_input_handle(name)
            input_tensor.reshape(batched_inputs[i].shape)
            input_tensor.copy_from_cpu(batched_inputs[i].copy())

        # do the inference
        predictor.run()

        # get inference process time cost
        if args.benchmark:
            autolog.times.stamp()

        # get out data from output tensor
        results = []
        # get out data from output tensor
        output_names = predictor.get_output_names()
        for i, name in enumerate(output_names):
            output_tensor = predictor.get_output_handle(name)
            output_data = output_tensor.copy_to_cpu()
            results.append(output_data)
        predict_label = np.argmax(results[0], 1)
        acc_batch = np.mean((predict_label == batch_label))
        acc.append(acc_batch)
        print('Batch action class Predict: ', predict_label,
              'Batch action class True: ', batch_label,
              'Batch Accuracy: ', acc_batch,
              'Batch sample Name: ', batch_sample_name)

        # get post process time cost
        if args.benchmark:
            autolog.times.end(stamp=True)
    print('Infer Mean Accuracy: ', np.mean(np.array(acc)))
    # report benchmark log if enabled
    if args.benchmark:
        autolog.report()


if __name__ == "__main__":
    main()
