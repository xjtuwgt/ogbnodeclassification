import sys
import os


PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import numpy as np
import time
import torch
from dgl import DGLGraph
from codes.gdthuge import GDTSampler
from codes.ioutils import EarlyStopping
from codes.ioutils import save_config, save_model, remove_models
from codes.utils import set_seeds, deep_dgl_graph_copy, graph_to_undirected, remove_self_loop_edges, reorginize_self_loop_edges
import argparse
from ogbdataloader.dglogbdataloader import DglNodePropPredDataset
import logging
from sklearn.metrics import roc_auc_score
from hugeGraphUtils.graphSampler import graph_k_neighbor_sampler, graph_ratio_neighbor_sampler

def parse_args(args=None):
    parser = argparse.ArgumentParser(description='GDT')
    if torch.cuda.is_available():
        parser.add_argument('--cuda', default=True, action='store_true', help='use GPU')
    else:
        parser.add_argument('--cuda', default=False, action='store_true', help='use GPU')
    parser.add_argument('--do_train', default=True, action='store_true')
    parser.add_argument("--data_path", type=str, default='../dataset')
    parser.add_argument("--dataset", type=str, default='ogbn_proteins')
    parser.add_argument("--epochs", type=int, default=300,
                        help="number of training epochs")
    parser.add_argument("--num_heads", type=int, default=4,
                        help="number of hidden attention heads")
    parser.add_argument("--num_layers", type=int, default=1,
                        help="number of hidden layers")
    parser.add_argument("--top_k", type=int, default=5,
                        help="top k selection")
    parser.add_argument("--project_dim", type=int, default=-1,
                        help="projection dimension")
    parser.add_argument("--num_hidden", type=int, default=16,
                        help="number of hidden units")
    parser.add_argument("--residual", action="store_true", default=True,
                        help="use residual connection")
    parser.add_argument("--in_drop", type=float, default=.5,
                        help="input feature dropout")
    parser.add_argument("--attn_drop", type=float, default=.25,
                        help="attention dropout")
    parser.add_argument("--edge_drop", type=float, default=.1,
                        help="edge dropout")
    parser.add_argument("--clip", type=float, default=1.0, help="grad_clip")
    parser.add_argument("--alpha", type=float, default=.6,
                        help="alpha")
    parser.add_argument("--hop_num", type=int, default=3,
                        help="hop number")
    parser.add_argument("--p_norm", type=int, default=0.5,
                        help="p_norm")
    parser.add_argument("--topk_type", type=str, default='local',
                        help="topk type")
    parser.add_argument("--sample_knn", type=int, default=100,
                        help="knn based sampler")
    parser.add_argument("--sample_ratio", type=float, default=0.1,
                        help="sample_ratio")
    parser.add_argument("--patience", type=int, default=300, help="patience")
    parser.add_argument('-save', '--save_path', default='../models/', type=str)
    parser.add_argument("--lr", type=float, default=0.0002,
                        help="learning rate")
    parser.add_argument("--lr_reduce_factor", type=float, default=0.5, help="Please give a value for lr_reduce_factor")
    parser.add_argument("--lr_schedule_patience", type=float, default=25, help="Please give a value for lr_reduce_patience")
    parser.add_argument('--weight_decay', type=float, default=1e-6,
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

def rocauc(y_pred, y_true):
    """
        compute ROC-AUC and AP score averaged across tasks
    """
    rocauc_list = []
    if torch is not None and isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    if torch is not None and isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            is_labeled = y_true[:, i] == y_true[:, i]
            rocauc_list.append(roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i]))
    if len(rocauc_list) == 0:
        raise RuntimeError("No positively labeled data available. Cannot compute ROC-AUC.")
    return sum(rocauc_list) / len(rocauc_list)


def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        predictions = model(features)
        predictions = predictions[mask]
        true_labels = labels[mask]
        return rocauc(predictions, true_labels)

def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''

    if args.do_train:
        log_file = os.path.join(args.save_path, 'train.log')
    else:
        log_file = os.path.join(args.save_path, 'test.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def preprocess(args):
    random_seed = args.seed
    set_seeds(random_seed)
    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    set_logger(args)
    logging.info("Model information...")
    for key, value in vars(args).items():
        logging.info('\t{} = {}'.format(key, value))

    model_folder_name = args2foldername(args)
    model_save_path = os.path.join(args.save_path, model_folder_name)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    save_config(args, model_save_path)
    logging.info('Model saving path: {}'.format(model_save_path))
    return model_save_path

def args2foldername(args):
    folder_name = args.dataset + 'lr_' + str(round(args.lr, 5)) + \
                 "lyer_" + str(args.num_layers) + 'hs_' + str(args.num_heads) + \
                 'ho_' + str(args.hop_num) + 'hi_' + str(args.num_hidden) + 'tk_' + str(args.top_k) + \
                 'pd_' + str(args.project_dim) + 'ind_' + str(round(args.in_drop, 4)) + \
                 'att_' + str(round(args.attn_drop, 4)) + 'ed_' + str(round(args.edge_drop, 4)) + 'alpha_' + \
                 str(round(args.alpha, 3)) + 'decay_' + str(round(args.edge_drop, 7))
    return folder_name

def ogb2dgl(args):
    data = DglNodePropPredDataset(name=args.dataset, root=args.data_path)
    graph, labels = data[0]
    def node_feature_aggregation(g: DGLGraph):
        start_time = time.time()
        g = g.local_var()  # the graph should be added a self-loop edge
        def send_edge_message(edges):
            return {'m_e': edges.data['feat']}
        def edge_feature_aggregation_func(nodes):
            edge_features = nodes.mailbox['m_e']
            edge_features = torch.mean(edge_features, 1)
            return {'node_feat': edge_features}
        g.register_reduce_func(edge_feature_aggregation_func)
        g.register_message_func(send_edge_message)
        g.update_all(message_func=send_edge_message, reduce_func=edge_feature_aggregation_func)
        print('Node feature aggregation in {} seconds'.format(time.time() - start_time))
        return g.ndata.pop('node_feat')
    features = node_feature_aggregation(g=graph)
    labels = labels.squeeze(dim=-1)
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
    number_class_labels = labels.shape[-1]
    ###
    graph.edata.pop('feat')
    ###
    return graph, features, labels, train_mask, valid_mask, test_mask, number_class_labels

def main(args):
    # load and preprocess dataset
    #+++++
    model_save_path = preprocess(args)
    #+++++
    graph, features, labels, train_mask, val_mask, test_mask, n_classes = ogb2dgl(args)

    graph = graph_k_neighbor_sampler(graph=graph, knn=args.sample_knn)
    # graph
    print(graph.number_of_nodes(), features.shape, labels.shape, train_mask.shape, val_mask.shape, test_mask.shape)
    num_feats = features.shape[1]
    n_edges = graph.number_of_edges()
    logging.info("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Feature dimension % d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes, num_feats,
           train_mask.int().sum().item(),
           val_mask.int().sum().item(),
           test_mask.int().sum().item()))

    cuda = args.cuda
    if cuda:
        features = features.cuda()
        labels = labels.cuda()
        train_mask = train_mask.cuda()
        val_mask = val_mask.cuda()
        test_mask = test_mask.cuda()

    g = deep_dgl_graph_copy(graph)
    #
    # graph_ratio_neighbor_sampler(graph=graph, sample_ratio=args.sample_ratio)
    # add self loop
    if args.self_loop == 1:
        g = remove_self_loop_edges(g)
        g.add_edges(g.nodes(), g.nodes())
        number_of_self_loops = g.number_of_nodes()
    else:
        zero_degree_idxes = g.in_degrees(np.arange(0, g.number_of_nodes())) == 0
        num_zero_degree = zero_degree_idxes.sum()
        if num_zero_degree > 0:
            zero_degree_nodes = torch.arange(0, g.number_of_nodes(), dtype=torch.long)[zero_degree_idxes]
            g.add_edges(zero_degree_nodes, zero_degree_nodes)
        g, number_of_self_loops = reorginize_self_loop_edges(graph=g)



    # print(number_of_self_loops, g.number_of_nodes())

    # print(g.in_degrees(np.arange(0, g.number_of_nodes())).float().median())
    n_edges = g.number_of_edges()
    # add edge ids
    edge_id = torch.arange(0, n_edges, dtype=torch.long)
    g.edata.update({'e_id': edge_id})
    if cuda:
        for key, value in g.ndata.items():
            g.ndata[key] = value.cuda()
        for key, value in g.edata.items():
            g.edata[key] = value.cuda()
    # # create model
    heads = [args.num_heads] * args.num_layers
    model = GDTSampler(g=g,
                num_layers=args.num_layers,
                input_dim=num_feats,
                project_dim=args.project_dim,
                hidden_dim=args.num_hidden,
                num_classes=n_classes,
                heads=heads,
                feat_drop=args.in_drop,
                attn_drop=args.attn_drop,
                alpha=args.alpha,
                hop_num=args.hop_num,
                top_k=args.top_k,
                topk_type=args.topk_type,
                edge_drop=args.edge_drop,
                knn_sampler=args.sample_knn,
                ratio_sampler=args.sample_ratio,
                number_self_loops=number_of_self_loops,
                self_loop=(args.self_loop==1),
                negative_slope=args.negative_slope)

    if cuda:
        model = model.cuda()
    logging.info(model)
    if args.early_stop:
        stopper = EarlyStopping(patience=100)
    if cuda:
        model.cuda()
    # loss_fcn = torch.nn.CrossEntropyLoss()
    loss_fcn = torch.nn.BCEWithLogitsLoss()
    weight_decay = args.weight_decay
    ##++++++++++++++++++++++++++++++++++++++++++++
    # use optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=weight_decay, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs, eta_min=1e-8)
    ##+++++++++++++++++++++++++++++++++++++++++++
    # initialize graph
    dur = []
    best_valid_acc = 0.0
    test_acc = 0.0
    patience_count = 0
    best_model_name = None
    for epoch in range(args.epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits = model(features)
        loss = loss_fcn(logits[train_mask], labels[train_mask].to(torch.float))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        scheduler.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        train_acc = rocauc(logits[train_mask], labels[train_mask])

        if args.fastmode:
            val_acc = rocauc(logits[val_mask], labels[val_mask])
        else:
            val_acc = evaluate(model, features, labels, val_mask)
            if args.early_stop:
                if stopper.step(val_acc, model):
                    break

        if val_acc >= best_valid_acc:
            best_valid_acc = val_acc
            acc = evaluate(model, features, labels, test_mask)
            # ++++++++++++++++++++++++++++++++++++++++
            model_name = str(epoch) + '_vacc_' + str(best_valid_acc) + '_tacc_' + str(acc) + '.pt'
            # model_path_name = os.path.join(model_save_path, model_name)
            # save_model(model, model_save_path=model_path_name, step=epoch)
            best_model_name = model_name
            # ++++++++++++++++++++++++++++++++++++++++
            test_acc = acc
            patience_count = 0
        else:
            patience_count = patience_count + 1

        logging.info("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | TrainAcc {:.4f} |"
              " ValAcc {:.4f} | ETputs(KTEPS) {:.2f}".
              format(epoch, np.mean(dur), loss.item(), train_acc,
                     val_acc, n_edges / np.mean(dur) / 1000))

        if patience_count >= args.patience:
            break

    logging.info('\n')
    logging.info('Best validation acc: {}\nBest test acc: {}'.format(best_valid_acc, test_acc))
    logging.info('Best model name: {}'.format(best_model_name))
    remove_models(model_save_path, best_model_name=best_model_name)

if __name__ == '__main__':
    main(parse_args())