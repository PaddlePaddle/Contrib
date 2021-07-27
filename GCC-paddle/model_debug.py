#!/usr/bin/env python
# encoding: utf-8
# File Name: train_graph_moco.py
# Author: Jiezhong Qiu
# Create Time: 2019/12/13 16:44
# test command: python model_debug.py --moco --nce-k 16384 --num-workers 1 --num-copies 1 --dataset h-index --gpu -1

import argparse
import copy
import os
import time
import warnings

import dgl
import numpy as np
import psutil
from visualdl import LogWriter
import paddorch as torch
import paddorch.nn as nn
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold


from gcc.contrastive.criterions import NCESoftmaxLoss, NCESoftmaxLossNS
from gcc.contrastive.memory_moco import MemoryMoCo
from gcc.datasets import (
    GRAPH_CLASSIFICATION_DSETS,
    GraphClassificationDataset,
    GraphClassificationDatasetLabeled,
    LoadBalanceGraphDataset,
    NodeClassificationDataset,
    NodeClassificationDatasetLabeled,
    worker_init_fn,
)
from gcc.datasets.data_util import batcher, labeled_batcher
from gcc.models import GraphEncoder
from gcc.utils.misc import AverageMeter, adjust_learning_rate, warmup_linear


def parse_option():

    # fmt: off
    parser = argparse.ArgumentParser("argument for training")

    parser.add_argument("--print-freq", type=int, default=10, help="print frequency")
    parser.add_argument("--tb-freq", type=int, default=250, help="tb frequency")
    parser.add_argument("--save-freq", type=int, default=1, help="save frequency")
    parser.add_argument("--batch-size", type=int, default=32, help="batch_size")
    parser.add_argument("--num-workers", type=int, default=1, help="num of workers to use")
    parser.add_argument("--num-copies", type=int, default=1, help="num of dataset copies that fit in memory")
    parser.add_argument("--num-samples", type=int, default=2000, help="num of samples per batch per worker")
    parser.add_argument("--epochs", type=int, default=100, help="number of training epochs")

    # optimization
    parser.add_argument("--optimizer", type=str, default='adam', choices=['sgd', 'adam', 'adagrad'], help="optimizer")
    parser.add_argument("--learning_rate", type=float, default=0.005, help="learning rate")
    parser.add_argument("--lr_decay_epochs", type=str, default="120,160,200", help="where to decay lr, can be a list")
    parser.add_argument("--lr_decay_rate", type=float, default=0.0, help="decay rate for learning rate")
    parser.add_argument("--beta1", type=float, default=0.9, help="beta1 for adam")
    parser.add_argument("--beta2", type=float, default=0.999, help="beta2 for Adam")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="weight decay")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
    parser.add_argument("--clip-norm", type=float, default=1.0, help="clip norm")

    # resume
    parser.add_argument("--resume", default="", type=str, metavar="PATH", help="path to latest checkpoint (default: none)")

    # augmentation setting
    parser.add_argument("--aug", type=str, default="1st", choices=["1st", "2nd", "all"])

    parser.add_argument("--exp", type=str, default="")

    # dataset definition
    parser.add_argument("--dataset", type=str, default="dgl", choices=["dgl", "wikipedia", "blogcatalog", "usa_airport", "brazil_airport", "europe_airport", "cora", "citeseer", "pubmed", "kdd", "icdm", "sigir", "cikm", "sigmod", "icde", "h-index-rand-1", "h-index-top-1", "h-index"] + GRAPH_CLASSIFICATION_DSETS)

    # model definition
    parser.add_argument("--model", type=str, default="gin", choices=["gat", "mpnn", "gin"])
    # other possible choices: ggnn, mpnn, graphsage ...
    parser.add_argument("--num-layer", type=int, default=5, help="gnn layers")
    parser.add_argument("--readout", type=str, default="avg", choices=["avg", "set2set"])
    parser.add_argument("--set2set-lstm-layer", type=int, default=3, help="lstm layers for s2s")
    parser.add_argument("--set2set-iter", type=int, default=6, help="s2s iteration")
    parser.add_argument("--norm", action="store_true", default=True, help="apply 2-norm on output feats")

    # loss function
    parser.add_argument("--nce-k", type=int, default=32)
    parser.add_argument("--nce-t", type=float, default=0.07)

    # random walk
    parser.add_argument("--rw-hops", type=int, default=256)
    parser.add_argument("--subgraph-size", type=int, default=128)
    parser.add_argument("--restart-prob", type=float, default=0.8)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--positional-embedding-size", type=int, default=32)
    parser.add_argument("--max-node-freq", type=int, default=16)
    parser.add_argument("--max-edge-freq", type=int, default=16)
    parser.add_argument("--max-degree", type=int, default=512)
    parser.add_argument("--freq-embedding-size", type=int, default=16)
    parser.add_argument("--degree-embedding-size", type=int, default=16)

    # specify folder
    parser.add_argument("--model-path", type=str, default="models", help="path to save model")
    parser.add_argument("--tb-path", type=str, default="tensorboard", help="path to tensorboard")
    parser.add_argument("--load-path", type=str, default=None, help="loading checkpoint at test time")

    # memory setting
    parser.add_argument("--moco", action="store_true", help="using MoCo (otherwise Instance Discrimination)")

    # finetune setting
    parser.add_argument("--finetune", action="store_true")

    parser.add_argument("--alpha", type=float, default=0.999, help="exponential moving average weight")

    # GPU setting
    parser.add_argument("--gpu", default=None, type=int, nargs='+', help="GPU id to use. -1: CPU only")

    # cross validation
    parser.add_argument("--seed", type=int, default=0, help="random seed.")
    parser.add_argument("--fold-idx", type=int, default=0, help="random seed.")
    parser.add_argument("--cv", action="store_true")
    # fmt: on

    opt = parser.parse_args()

    iterations = opt.lr_decay_epochs.split(",")
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    return opt


def option_update(opt):
    opt.model_name = "{}_moco_{}_{}_{}_layer_{}_lr_{}_decay_{}_bsz_{}_hid_{}_samples_{}_nce_t_{}_nce_k_{}_rw_hops_{}_restart_prob_{}_aug_{}_ft_{}_deg_{}_pos_{}_momentum_{}".format(
        opt.exp,
        opt.moco,
        opt.dataset,
        opt.model,
        opt.num_layer,
        opt.learning_rate,
        opt.weight_decay,
        opt.batch_size,
        opt.hidden_size,
        opt.num_samples,
        opt.nce_t,
        opt.nce_k,
        opt.rw_hops,
        opt.restart_prob,
        opt.aug,
        opt.finetune,
        opt.degree_embedding_size,
        opt.positional_embedding_size,
        opt.alpha,
    )

    if opt.load_path is None:
        opt.model_folder = os.path.join(opt.model_path, opt.model_name)
        if not os.path.isdir(opt.model_folder):
            os.makedirs(opt.model_folder)
    else:
        opt.model_folder = opt.load_path

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)
    return opt


def moment_update(model, model_ema, m):
    """ model_ema = m * model_ema + (1 - m) model """
    for p1, p2 in zip(model.parameters(), model_ema.parameters()):
        new_p2=(p2.detach() *float(m) )+(float(1 - m)*p1.detach() )
        p2.set_value(new_p2)


def clip_grad_norm(params, max_norm):
    """Clips gradient norm."""
    if max_norm > 0:
        return [torch.nn.utils.clip_by_norm(p, max_norm)  for p in params if p.grad is not None]
    else:
        return torch.sqrt(
            sum(p.grad.data.norm() ** 2 for p in params if p.grad is not None)
        )


# def main(args, trial):
def main(args):
    dgl.random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.gpu>=0:
        torch.cuda.manual_seed(args.seed)
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location="cpu")
            pretrain_args = checkpoint["opt"]
            pretrain_args.fold_idx = args.fold_idx
            pretrain_args.gpu = args.gpu
            pretrain_args.finetune = args.finetune
            pretrain_args.resume = args.resume
            pretrain_args.cv = args.cv
            pretrain_args.dataset = args.dataset
            pretrain_args.epochs = args.epochs
            pretrain_args.num_workers = args.num_workers
            if args.dataset in GRAPH_CLASSIFICATION_DSETS:
                # HACK for speeding up finetuning on graph classification tasks
                pretrain_args.num_workers = 0
            pretrain_args.batch_size = args.batch_size
            args = pretrain_args
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    args = option_update(args)
    print(args)
    if args.gpu>=0:
        assert args.gpu is not None and torch.cuda.is_available()
        print("Use GPU: {} for training".format(args.gpu))
    assert args.positional_embedding_size % 2 == 0
    print("setting random seeds")

    mem = psutil.virtual_memory()
    print("before construct dataset", mem.used / 1024 ** 3)
    if args.finetune:
        if args.dataset in GRAPH_CLASSIFICATION_DSETS:
            dataset = GraphClassificationDatasetLabeled(
                dataset=args.dataset,
                rw_hops=args.rw_hops,
                subgraph_size=args.subgraph_size,
                restart_prob=args.restart_prob,
                positional_embedding_size=args.positional_embedding_size,
            )
            labels = dataset.dataset.data.y.tolist()
        else:
            dataset = NodeClassificationDatasetLabeled(
                dataset=args.dataset,
                rw_hops=args.rw_hops,
                subgraph_size=args.subgraph_size,
                restart_prob=args.restart_prob,
                positional_embedding_size=args.positional_embedding_size,
            )
            labels = dataset.data.y.argmax(dim=1).tolist()

        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=args.seed)
        idx_list = []
        for idx in skf.split(np.zeros(len(labels)), labels):
            idx_list.append(idx)
        assert (
            0 <= args.fold_idx and args.fold_idx < 10
        ), "fold_idx must be from 0 to 9."
        train_idx, test_idx = idx_list[args.fold_idx]
        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        valid_dataset = torch.utils.data.Subset(dataset, test_idx)

    elif args.dataset == "dgl":
        train_dataset = LoadBalanceGraphDataset(
            rw_hops=args.rw_hops,
            restart_prob=args.restart_prob,
            positional_embedding_size=args.positional_embedding_size,
            num_workers=args.num_workers,
            num_samples=args.num_samples,
            dgl_graphs_file="./data/small.bin",
            num_copies=args.num_copies,
        )
    else:
        if args.dataset in GRAPH_CLASSIFICATION_DSETS:
            train_dataset = GraphClassificationDataset(
                dataset=args.dataset,
                rw_hops=args.rw_hops,
                subgraph_size=args.subgraph_size,
                restart_prob=args.restart_prob,
                positional_embedding_size=args.positional_embedding_size,
            )
        else:
            train_dataset = NodeClassificationDataset(
                dataset=args.dataset,
                rw_hops=args.rw_hops,
                subgraph_size=args.subgraph_size,
                restart_prob=args.restart_prob,
                positional_embedding_size=args.positional_embedding_size,
            )

    mem = psutil.virtual_memory()
    print("before construct dataloader", mem.used / 1024 ** 3)
    train_loader = torch.utils.data.graph.Dataloader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        collate_fn= labeled_batcher() if args.finetune else batcher(),
        shuffle=True if args.finetune else False,
        num_workers=args.num_workers,
        worker_init_fn=None
        if args.finetune or args.dataset != "dgl"
        else worker_init_fn,
    )
    if args.finetune:
        valid_loader = torch.utils.data.DataLoader(
            dataset=valid_dataset,
            batch_size=args.batch_size,
            collate_fn=labeled_batcher(),
            num_workers=args.num_workers,
        )
    mem = psutil.virtual_memory()
    print("before training", mem.used / 1024 ** 3)

    # create model and optimizer
    # n_data = train_dataset.total
    n_data = None
    import gcc.models.graph_encoder
    gcc.models.graph_encoder.final_dropout=0 ##disable dropout
    model, model_ema = [
        GraphEncoder(
            positional_embedding_size=args.positional_embedding_size,
            max_node_freq=args.max_node_freq,
            max_edge_freq=args.max_edge_freq,
            max_degree=args.max_degree,
            freq_embedding_size=args.freq_embedding_size,
            degree_embedding_size=args.degree_embedding_size,
            output_dim=args.hidden_size,
            node_hidden_dim=args.hidden_size,
            edge_hidden_dim=args.hidden_size,
            num_layers=args.num_layer,
            num_step_set2set=args.set2set_iter,
            num_layer_set2set=args.set2set_lstm_layer,
            norm=args.norm,
            gnn_model=args.model,
            degree_input=True,
        )
        for _ in range(2)
    ]

    # copy weights from `model' to `model_ema'
    if args.moco:
        moment_update(model, model_ema, 0)

    # set the contrast memory and criterion
    contrast = MemoryMoCo(
        args.hidden_size, n_data, args.nce_k, args.nce_t, use_softmax=True
    )
    if args.gpu>=0:
        contrast=contrast.cuda(args.gpu)

    if args.finetune:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = NCESoftmaxLoss() if args.moco else NCESoftmaxLossNS()
        if args.gpu>=0:
            criterion = criterion.cuda(args.gpu)
    if args.gpu >= 0:
        model = model.cuda(args.gpu)
        model_ema = model_ema.cuda(args.gpu)

    if args.finetune:
        output_layer = nn.Linear(
            in_features=args.hidden_size, out_features=dataset.num_classes
        )
        if args.gpu >= 0:
            output_layer = output_layer.cuda(args.gpu)
        output_layer_optimizer = torch.optim.Adam(
            output_layer.parameters(),
            lr=args.learning_rate,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
        )

        def clear_bn(m):
            classname = m.__class__.__name__
            if classname.find("BatchNorm") != -1:
                m.reset_running_stats()

        model.apply(clear_bn)

    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "adagrad":
        optimizer = torch.optim.Adagrad(
            model.parameters(),
            lr=args.learning_rate,
            lr_decay=args.lr_decay_rate,
            weight_decay=args.weight_decay,
        )
    else:
        raise NotImplementedError

    # optionally resume from a checkpoint
    args.start_epoch = 1
    if True:
        # print("=> loading checkpoint '{}'".format(args.resume))
        # checkpoint = torch.load(args.resume, map_location="cpu")
        import torch as th
        checkpoint = th.load("torch_models/ckpt_epoch_100.pth",map_location=th.device('cpu'))
        torch_input_output_grad=th.load("torch_models/torch_input_output_grad.pt", map_location=th.device('cpu'))
        from paddorch.convert_pretrain_model import load_pytorch_pretrain_model
        print("loading.............. model")
        paddle_state_dict = load_pytorch_pretrain_model(model, checkpoint["model"])
        model.load_state_dict(paddle_state_dict)
        print("loading.............. contrast")
        paddle_state_dict2 = load_pytorch_pretrain_model(contrast, checkpoint["contrast"])
        contrast.load_state_dict(paddle_state_dict2)
        print("loading.............. model_ema")
        paddle_state_dict3 = load_pytorch_pretrain_model(model_ema, checkpoint["model_ema"])
        if args.moco:
            model_ema.load_state_dict(paddle_state_dict3)

        print(
            "=> loaded successfully '{}' (epoch {})".format(
                args.resume, checkpoint["epoch"]
            )
        )
        del checkpoint
        if args.gpu >= 0:
            torch.cuda.empty_cache()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate*0.1,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
        )
        for _ in range(1):
            graph_q, graph_k=train_dataset[0]
            graph_q2, graph_k2 = train_dataset[1]
            graph_q, graph_k = dgl.batch([graph_q,graph_q2]), dgl.batch([graph_k,graph_k2])

            input_output_grad = []
            input_output_grad.append([graph_q, graph_k])
            model.train()
            model_ema.eval()

            feat_q = model(graph_q)
            with torch.no_grad():
                feat_k = model_ema(graph_k)

            out = contrast(feat_q, feat_k)
            loss = criterion(out )
            optimizer.zero_grad()
            loss.backward()
            input_output_grad.append([feat_q,out,loss])
            print("loss:",loss.numpy())
            optimizer.step()
            moment_update(model, model_ema, args.alpha)
        print("max diff feat_q:",np.max(np.abs(torch_input_output_grad[1][0].detach().numpy() -feat_q.numpy())))
        print("max diff out:", np.max(np.abs(torch_input_output_grad[1][1].detach().numpy() - out.numpy())))
        print("max diff loss:", np.max(np.abs(torch_input_output_grad[1][2].detach().numpy() - loss.numpy())))

        name2grad=dict()
        for name,p in dict(model.named_parameters()).items():
            if p.grad is not None:
                name2grad[name]=p.grad
                torch_grad=torch_input_output_grad[2][name].numpy()

                if "linear" in name and "weight" in name:
                    torch_grad=torch_grad.T
                max_grad_diff=np.max(np.abs(p.grad -torch_grad))
                print("max grad diff:",name,max_grad_diff)
        input_output_grad.append(name2grad)




if __name__ == "__main__":

    warnings.simplefilter("once", UserWarning)
    args = parse_option()

    if args.cv:
        gpus = args.gpu

        def variant_args_generator():
            for fold_idx in range(10):
                args.fold_idx = fold_idx
                args.num_workers = 0
                args.gpu = gpus[fold_idx % len(gpus)]
                yield copy.deepcopy(args)

        # f1 = Parallel(n_jobs=5)(
        #     delayed(main)(args) for args in variant_args_generator()
        # )
        f1 = [main(args) for args in variant_args_generator()]
        print(f1)
        print(f"Mean = {np.mean(f1)}; Std = {np.std(f1)}")
    else:
        args.gpu = args.gpu[0]
        main(args)
    # import optuna
    # def objective(trial):
    #     args.epochs = 50
    #     args.learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
    #     args.weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-3)
    #     args.restart_prob = trial.suggest_uniform('restart_prob', 0.5, 1)
    #     # args.alpha = 1 - trial.suggest_loguniform('alpha', 1e-4, 1e-2)
    #     return main(args, trial)

    # study = optuna.load_study(study_name='cat_prone', storage="sqlite:///example.db")
    # study.optimize(objective, n_trials=20)
