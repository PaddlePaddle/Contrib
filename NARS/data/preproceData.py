import os
import numpy as np
import torch
import dgl
import dgl.function as fn

###############################################################################
# Loading Relation Subsets
###############################################################################

def read_relation_subsets(fname):
    print("Reading Relation Subsets:")
    rel_subsets = []
    with open(fname) as f:
        for line in f:
            relations = line.strip().split(',')
            rel_subsets.append(relations)
            print(relations)
    return rel_subsets


###############################################################################
# Generate multi-hop neighbor-averaged feature for each relation subset
###############################################################################
def load_data(device):
    return load_mag(device)

def load_mag(device):
    from ogb.nodeproppred import DglNodePropPredDataset
    path = 'embedding/'
    home_dir = './'
    dataset = DglNodePropPredDataset(
        name="ogbn-mag", root=os.path.join(home_dir, "ogb", "dataset"))
    g, labels = dataset[0]
    splitted_idx = dataset.get_idx_split()
    train_nid = splitted_idx["train"]['paper']
    val_nid = splitted_idx["valid"]['paper']
    test_nid = splitted_idx["test"]['paper']
    features = g.nodes['paper'].data['feat']
    author_emb = torch.load(os.path.join(path, "author.pt")).float()
    topic_emb = torch.load(os.path.join(path, "field_of_study.pt")).float()
    institution_emb = torch.load(os.path.join(path, "institution.pt")).float()
    g.nodes["author"].data["feat"] = author_emb.to(device)
    g.nodes["institution"].data["feat"] = institution_emb.to(device)
    g.nodes["field_of_study"].data["feat"] = topic_emb.to(device)
    g.nodes["paper"].data["feat"] = features.to(device)
    paper_dim = g.nodes["paper"].data["feat"].shape[1]
    author_dim = g.nodes["author"].data["feat"].shape[1]
    if paper_dim != author_dim:
        paper_feat = g.nodes["paper"].data.pop("feat")
        rand_weight = torch.Tensor(paper_dim, author_dim).uniform_(-0.5, 0.5)
        g.nodes["paper"].data["feat"] = torch.matmul(paper_feat, rand_weight.to(device))
        print(f"Randomly project paper feature from dimension {paper_dim} to {author_dim}")

    labels = labels['paper'].to(device).squeeze()
    n_classes = int(labels.max() - labels.min()) + 1
    train_nid, val_nid, test_nid = np.array(train_nid), np.array(val_nid), np.array(test_nid)
    return g, labels, n_classes, train_nid, val_nid, test_nid
def gen_rel_subset_feature(g, rel_subset, args, device):
    """
    Build relation subgraph given relation subset and generate multi-hop
    neighbor-averaged feature on this subgraph
    """
    if args.cpu_preprocess:
        device = "cpu"
    new_edges = {}
    ntypes = set()
    print('etype' + '*'*20)
    for etype in rel_subset:
        
        print('etype', etype)
        stype, _, dtype = g.to_canonical_etype(etype)
        print('to_canonical_etype', g.to_canonical_etype(etype))
        src, dst = g.all_edges(etype=etype)
        src = src.numpy()
        dst = dst.numpy()
        new_edges[(stype, etype, dtype)] = (src, dst)
        new_edges[(dtype, etype + "_r", stype)] = (dst, src)
        ntypes.add(stype)
        ntypes.add(dtype)
    new_g = dgl.heterograph(new_edges)
    
    # set node feature and calc deg
    for ntype in ntypes:
        num_nodes = new_g.number_of_nodes(ntype)
        if num_nodes < g.nodes[ntype].data["feat"].shape[0]:
            new_g.nodes[ntype].data["hop_0"] = g.nodes[ntype].data["feat"][:num_nodes, :]
        else:
            new_g.nodes[ntype].data["hop_0"] = g.nodes[ntype].data["feat"]
        deg = 0
        for etype in new_g.etypes:
            _, _, dtype = new_g.to_canonical_etype(etype)
            if ntype == dtype:
                deg = deg + new_g.in_degrees(etype=etype)
        norm = 1.0 / deg.float()
        norm[torch.isinf(norm)] = 0
        new_g.nodes[ntype].data["norm"] = norm.view(-1, 1).to(device)

    res = []

    # compute k-hop feature
    for hop in range(1, args.R + 1):
        ntype2feat = {}
        for etype in new_g.etypes:
            stype, _, dtype = new_g.to_canonical_etype(etype)
            new_g[etype].update_all(fn.copy_u(f'hop_{hop-1}', 'm'), fn.sum('m', 'new_feat'))
            new_feat = new_g.nodes[dtype].data.pop("new_feat")
            assert("new_feat" not in new_g.nodes[stype].data)
            if dtype in ntype2feat:
                ntype2feat[dtype] += new_feat
            else:
                ntype2feat[dtype] = new_feat
        for ntype in new_g.ntypes:
            assert ntype in ntype2feat  # because subgraph is not directional
            feat_dict = new_g.nodes[ntype].data
            old_feat = feat_dict.pop(f"hop_{hop-1}")
            if ntype == "paper":
                res.append(old_feat.cpu())
            feat_dict[f"hop_{hop}"] = ntype2feat.pop(ntype).mul_(feat_dict["norm"])

    res.append(new_g.nodes["paper"].data.pop(f"hop_{args.R}").cpu())
    return res
def preprocess_features(g, rel_subsets, args, device):
    # pre-process heterogeneous graph g to generate neighbor-averaged features
    # for each relation subsets
    num_paper, feat_size = g.nodes["paper"].data["feat"].shape
    new_feats = [torch.zeros([num_paper, len(rel_subsets), feat_size]) for _ in range(args.R + 1)]
    print("Start generating features for each sub-metagraph:")
    for subset_id, subset in enumerate(rel_subsets):
        print(subset)
        feats = gen_rel_subset_feature(g, subset, args, device)
        for i in range(args.R + 1):
            feat = feats[i]
            new_feats[i][:feat.shape[0], subset_id, :] = feat
        feats = None
    return new_feats
if __name__ == '__main__':
    import paddle
    import argparse
    parser = argparse.ArgumentParser(description="Neighbor-Averaging over Relation Subgraphs")
    parser.add_argument("--num-epochs", type=int, default=1000)
    parser.add_argument("--num-hidden", type=int, default=256)
    parser.add_argument("--R", type=int, default=5,
                        help="number of hops")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dataset", type=str, default="oag")
    parser.add_argument("--data-dir", type=str, default=None, help="path to dataset, only used for OAG")
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=0)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=50000)
    parser.add_argument("--eval-batch-size", type=int, default=250000,
                        help="evaluation batch size, -1 for full batch")
    parser.add_argument("--ff-layer", type=int, default=2,
                        help="number of feed-forward layers")
    parser.add_argument("--input-dropout", action="store_true")
    parser.add_argument("--use-emb", default='embedding', type=str)
    parser.add_argument("--seed", type=int, default=None )
    parser.add_argument("--cpu-preprocess", action="store_true",
                        help="Preprocess on CPU")
    args = parser.parse_args([])
    print('-------------load data------------------')
    data, labels, num_classes, train_nid, val_nid, test_nid = load_mag('cpu')
    print(labels, num_classes, train_nid, val_nid, test_nid)
    np.save('data/lables.npy', labels.numpy())
    np.save('data/num_classes.npy', num_classes)
    np.save('data/train_nid.npy', train_nid)
    np.save('data/val_nid.npy', val_nid)
    np.save('data/test_nid.npy', test_nid)
    rel_subsets = read_relation_subsets('./sample/8_mag_rand_subsets')
    print('-------------preprocess data------------------')
    with torch.no_grad():
        feats = preprocess_features(data, rel_subsets, args, 'cpu')
    print('-------------save data------------------')
    for i in range(len(feats)):
        np.save(f'new_feature/feat{i}.npy', feats[i].numpy())
    