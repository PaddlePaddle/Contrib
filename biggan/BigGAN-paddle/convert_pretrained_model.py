import paddorch
from paddorch.convert_pretrain_model import load_pytorch_pretrain_model
from glob import glob
import utils
from paddle import fluid
import os
import numpy as np
input_weight_folder="best_weigths/BigGAN_C10_seed0_Gch64_Dch64_bs128_nDs4_Glr2.0e-04_Dlr2.0e-04_Gnlrelu_Dnlrelu_GinitN02_DinitN02_ema"

import json

def get_pytorch_G_model(config):
    from torch_BigGAN import Generator
    from torch.nn import  ReLU
    config2=config.copy()
    config2['D_activation']=ReLU()
    config2['G_activation'] = ReLU()
    return Generator(**config2)



if __name__ == '__main__':
  from paddle import fluid
  place=fluid.CUDAPlace(0)
  with fluid.dygraph.guard(place=place):
        config=json.load( open("c10_config.json", 'r'))

        config['G_activation'] = utils.activation_dict[config['G_nl']]
        config['D_activation'] = utils.activation_dict[config['D_nl']]
        # By default, skip init if resuming training.
        if config['resume']:
            print('Skipping initialization for training resumption...')
            config['skip_init'] = True

        config = utils.update_config_roots(config)
        device = 'cuda'

        # Seed RNG
        utils.seed_rng(config['seed'])

        # Prepare root folders if necessary
        utils.prepare_root(config)

        # Import the model--this line allows us to dynamically select different files.
        model = __import__(config['model'])


        for torch_fn in glob("%s/*pth"%input_weight_folder):
            if "optim" in torch_fn:
                continue # skip optimizer file

            if "best3" not in torch_fn:
                continue
            import torch as pytorch
            torch_state_dict= pytorch.load(torch_fn)

            # Next, build the model
            print(torch_fn)
            out_fn = torch_fn.replace(".pth", ".pdparams")
            if os.path.basename(torch_fn).startswith("G"):
                G = model.Generator(**config)
                torch_G=get_pytorch_G_model(config)
                torch_G.load_state_dict(torch_state_dict)
                load_pytorch_pretrain_model(G,torch_state_dict)
                z=np.zeros((1,128))
                y=np.ones((1,1))
                import torch as pytorch
                torch_X=torch_G(pytorch.Tensor(z),pytorch.LongTensor(y))
                X=G(paddorch.Tensor(z).astype("float32"),paddorch.Tensor(y).astype("int64"))
                print(torch_X.detach().numpy().mean(),X.detach().numpy().mean())
                print("saved file:",out_fn)
                paddorch.save(G.state_dict(),out_fn)
            elif   os.path.basename(torch_fn).startswith("D"):
                D = model.Discriminator(**config)
                load_pytorch_pretrain_model(D,torch_state_dict)
                paddorch.save(D.state_dict(),out_fn)
                print("saved file:", out_fn)
            else: ##state_dict
                torch_state_dict['config']['D_activation'] =paddorch.nn.ReLU().state_dict()
                torch_state_dict['config']['G_activation'] = paddorch.nn.ReLU().state_dict()
                fluid.dygraph.save_dygraph(torch_state_dict,out_fn)
                os.system("mv %s.pdopt %s"%(out_fn,out_fn))


