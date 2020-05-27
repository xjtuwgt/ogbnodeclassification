import sys
import os


PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import numpy as np
import time
import torch
import argparse
from ogbdataloader.dglogbdataloader import DglNodePropPredDataset
import logging

def parse_args(args=None):
    parser = argparse.ArgumentParser(description='GDT')
    parser.add_argument('--cuda', default=False, action='store_true', help='use GPU')
    parser.add_argument('--do_train', default=True, action='store_true')
    parser.add_argument("--datafolder", type=str, default='../dataset')
    parser.add_argument("--dataset", type=str, default='ogbn-arxiv')
    parser.add_argument("--epochs", type=int, default=10,
                        help="number of training epochs")
    parser.add_argument("--num_heads", type=int, default=8,
                        help="number of hidden attention heads")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="number of hidden layers")
    parser.add_argument("--top_k", type=int, default=5,
                        help="top k selection")
    parser.add_argument("--project_dim", type=int, default=-1,
                        help="projection dimension")
    parser.add_argument("--num_hidden", type=int, default=256,
                        help="number of hidden units")
    parser.add_argument("--residual", action="store_true", default=True,
                        help="use residual connection")
    parser.add_argument("--in_drop", type=float, default=.5,
                        help="input feature dropout")
    parser.add_argument("--attn_drop", type=float, default=.6,
                        help="attention dropout")
    parser.add_argument("--edge_drop", type=float, default=.1,
                        help="edge dropout")
    parser.add_argument("--clip", type=float, default=1.0, help="grad_clip")
    parser.add_argument("--alpha", type=float, default=.2,
                        help="alpha")
    parser.add_argument("--hop_num", type=int, default=3,
                        help="hop number")
    parser.add_argument("--p_norm", type=int, default=0.5,
                        help="p_norm")
    parser.add_argument("--topk_type", type=str, default='local',
                        help="topk type")
    parser.add_argument("--patience", type=int, default=300, help="patience")
    parser.add_argument('-save', '--save_path', default='../models/', type=str)
    parser.add_argument("--lr", type=float, default=0.0005,
                        help="learning rate")
    parser.add_argument("--lr_reduce_factor", type=float, default=0.5, help="Please give a value for lr_reduce_factor")
    parser.add_argument("--lr_schedule_patience", type=float, default=25, help="Please give a value for lr_reduce_patience")
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help="weight decay")
    parser.add_argument('--negative_slope', type=float, default=0.2,
                        help="the negative slope of leaky relu")
    parser.add_argument('--self_loop', default=1, type=int, help='whether self-loop')
    parser.add_argument('--early-stop', action='store_true', default=False,
                        help="indicates whether to use early stop or not")
    parser.add_argument('--fastmode', action="store_true", default=False,
                        help="skip re-evaluate the validation set")
    parser.add_argument('--seed', type=int, default=2019,
                        help="random seed")
    args = parser.parse_args(args)
    return args

def ogb2dgl(args):
    data = DglNodePropPredDataset(name=args.dataset, root=args.datafolder)
    graph, labels = data[0]
    labels = labels.squeeze(dim=-1)
    features, node_year = graph.ndata['feat'], graph.ndata['node_year']
    number_nodes = graph.number_of_nodes()
    split_idx = data.get_idx_split()
    train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
    train_mask, valid_mask, test_mask = torch.zeros(number_nodes), torch.zeros(number_nodes), torch.zeros(number_nodes)
    train_mask[train_idx] = 1
    valid_mask[valid_idx] = 1
    test_mask[test_idx] = 1
    if hasattr(torch, 'BoolTensor'):
        train_mask = torch.BoolTensor(train_mask.numpy())
        valid_mask = torch.BoolTensor(valid_mask.numpy())
        test_mask = torch.BoolTensor(test_mask.numpy())
    else:
        train_mask = torch.ByteTensor(train_mask.numpy())
        valid_mask = torch.ByteTensor(valid_mask.numpy())
        test_mask = torch.ByteTensor(test_mask.numpy())
    number_class_labels = torch.unique(labels).shape[0]
    return graph, features, labels, train_mask, valid_mask, test_mask, number_class_labels

def main(args):
    # load and preprocess dataset
    graph, features, labels, train_mask, val_mask, test_mask, n_classes = ogb2dgl(args)
    print(graph.number_of_nodes(), features.shape, labels.shape, train_mask.shape, val_mask.shape, test_mask.shape)

if __name__ == '__main__':
    main(parse_args())