import paddle.fluid as fluid 
from paddle.fluid.executor import Executor
import argparse
from reader import construct_sample, train_reader, val_reader, Y_seq 

class RetDict(object):
    """doc"""
    pass
class SeqModel(object):
    def __init__(self, seq_num, batch_size, hidden_size):
       self.name = "seq2seq"
       self.seq_num = seq_num
       self.hidden_size = hidden_size
       self.batch_size = batch_size
       self.dropout_prob = 0.5
       self.feat_dim = 74

    def _build_data(self):
        self.feat = fluid.layers.data(name="feat", shape=[self.feat_dim], dtype='float32', lod_level=1)
        self.label = fluid.layers.data(name="label", shape=[Y_seq, self.feat_dim], dtype="float32")
        self.lod_message = fluid.layers.data(name="lod", shape=[1], dtype="int32")

    def _forward(self):
        self.lod_feat = fluid.layers.lod_reset(self.feat, self.lod_message) 
        forward_proj = fluid.layers.fc(input=self.lod_feat, size=self.hidden_size * 4, bias_attr=False)
         
        forward, cell = fluid.layers.dynamic_lstm(input=forward_proj, size=self.hidden_size * 4, use_peepholes=True)
        forward = fluid.layers.expand(x=forward, expand_times=[Y_seq ,1])
        forwards = fluid.layers.split(forward, num_or_sections=Y_seq, dim=0)
        concat_forwards = [] 
        for i in range(0, Y_seq):
            forwards[i] = fluid.layers.lod_reset(forwards[i], self.lod_message)
            input_lstm  = fluid.layers.fc(forwards[i], size=self.hidden_size*4, bias_attr=False)
            forward_1, _  = fluid.layers.dynamic_lstmp(input=input_lstm, name="f_forward_%d"%(i), size=self.hidden_size*4, proj_size=self.feat_dim,\
                 cell_activation="tanh",
                 proj_activation="tanh",
                 use_peepholes=False)
            forward_2, _  = fluid.layers.dynamic_lstmp(input=input_lstm, name="r_forward_%d"%(i), size=self.hidden_size*4, proj_size=self.feat_dim,\
                is_reverse=True,
                cell_activation="tanh",
                proj_activation="tanh",
                use_peepholes=False)
            last_forward_1 = fluid.layers.sequence_last_step(forward_1)
            last_forward_2 = fluid.layers.sequence_last_step(forward_2)
            
            concat_forward = (last_forward_1 + last_forward_1)/2
            concat_forwards.append(concat_forward)
            
        concat_prediction = fluid.layers.concat(concat_forwards, axis=0)
        self.last_predict = fluid.layers.reshape(concat_prediction, shape=[-1, Y_seq, self.feat_dim])
        self.last_predict = fluid.layers.abs(self.last_predict)

    def _compute_loss(self):
        self.loss = fluid.layers.mse_loss(input=self.last_predict, label=self.label) 
        
    def build_graph(self):
        self._build_data()
        self._forward()
        self._compute_loss()
        # return model message 
        ret_dict = RetDict()
        ret_dict.feed_list = [self.feat, self.label, self.lod_message]
        ret_dict.fetch_list = [self.loss, self.last_predict]
        ret_dict.loss = self.loss
        ret_dict.last_predict = self.last_predict
        return ret_dict

def main(args):

    # construct the sample 
    input, output, data_size = construct_sample(args)
    # construct the train program
    train_program = fluid.Program()
    startup_program = fluid.Program()

    with fluid.program_guard(train_program, startup_program):
         seq2seq_model = SeqModel(args.seq_num, args.batch_size, args.hidden_size)
         ret_dict = seq2seq_model.build_graph()
    val_program = train_program.clone()
    with fluid.program_guard(train_program, startup_program):
         optimizer = fluid.optimizer.Adam(args.lr)
         optimizer.minimize(ret_dict.loss) 
    places = fluid.cuda_places() if args.use_cuda else fluid.cpu_places()

    train_loader = fluid.io.DataLoader.from_generator(feed_list=ret_dict.feed_list, capacity=3, iterable=True)
    train_loader.set_batch_generator(train_reader(input, output, data_size, args.batch_size), places=places)

    exe = Executor(places[0])
    exe.run(startup_program)

    # train stage:use data_loader as reader 
    for _ in range(args.epoch):
        for data in train_loader():
            results = exe.run(train_program, feed=data, fetch_list=ret_dict.fetch_list)
            print("train process loss:{}".format(results[0]))
    # save the model for inferenceing
    with fluid.program_guard(train_program, startup_program):
        fluid.io.save_inference_model(dirname="./model", feeded_var_names=['feat', 'lod'], \
            target_vars=[ret_dict.last_predict], executor=exe, export_for_deployment=True)

    # val stage: use data_loader as reader  
    val_loader = fluid.io.DataLoader.from_generator(feed_list=ret_dict.feed_list, capacity=3, iterable=True)
    val_loader.set_batch_generator(val_reader(input, output, data_size, output.shape[0]), places=places)
    for _ in range(1):
        for data in train_loader:
            results = exe.run(train_program, feed=data, fetch_list=ret_dict.fetch_list)
            print("val process loss:{}".format(results[0]))

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='seq2seq')
    parser.add_argument("--use_cuda", action='store_true', help="use_cuda")
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=40)
    parser.add_argument("--seq_num", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.1)
    args = parser.parse_args()
    main(args)
