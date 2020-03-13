import paddle.fluid as fluid 
from paddle.fluid.executor import Executor
import pandas as pd 
import numpy as np
import argparse
from reader_batch import construct_sample, test_reader
from reader_batch import Y_seq, feature_num
import os

def output_final_result(prediction, idx):
    city = pd.read_csv("./data/crawl_list.csv")
    city_list = list(city['city'])
    output = pd.read_csv("./data/output.csv",index_col=0)
    out_max = np.array(output.max())[1:]
    out_max = np.tile(out_max,Y_seq).reshape(Y_seq,feature_num)
    out_min = np.array(output.min())[1:]
    out_min = np.tile(out_min,Y_seq).reshape(Y_seq,feature_num)
    sub = out_max-out_min
    y2pre = np.reshape(prediction, (Y_seq ,feature_num))
    prediction = pd.DataFrame(np.around(y2pre*sub)) 
    prediction.columns = city_list[1:]
    prediction.index = prediction.index + 1
    out_folder = 'output'
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    prediction.to_csv(os.path.join(out_folder,"prediction_"+idx+".csv"))
    
def main(args):
    input, output, data_size = construct_sample(args)
    places = fluid.cuda_places() if args.use_cuda else fluid.cpu_places()
    exe = Executor(places[0])
    [inference_program, feed_target_names, fetch_targets] = (fluid.io.load_inference_model(dirname="./model", executor=exe))
    feats, index, lod = test_reader()
    results = []
    for i, feat in enumerate(feats):
        result = exe.run(inference_program, 
        feed={feed_target_names[0]: feat, feed_target_names[1]: lod}, 
        fetch_list=fetch_targets)
        output_final_result(result[0], index[i])

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='seq2seq')
    parser.add_argument("--use_cuda", action='store_true', help="use_cuda")
    parser.add_argument("--epoch", type=int, default=300)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=40)
    parser.add_argument("--seq_num", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.01)
    args = parser.parse_args()
    main(args)
