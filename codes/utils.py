import torch
import os
import dgl
import numpy as np
import random
from time import time
from dgl import DGLGraph

def set_seeds(seed):
    "set random seeds"
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True
    dgl.random.seed(seed)

def deep_dgl_graph_copy(graph: DGLGraph):
    start = time()
    copy_graph = DGLGraph()
    copy_graph.add_nodes(graph.number_of_nodes())
    graph_edges = graph.edges()
    copy_graph.add_edges(graph_edges[0], graph_edges[1])
    for key, value in graph.edata.items():
        copy_graph.edata[key] = value
    for key, value in graph.ndata.items():
        copy_graph.ndata[key] = value
    print('Graph copy take {:.2f} seconds'.format(time() - start))
    return copy_graph

def remove_self_loop_edges(graph: DGLGraph):
    g_src, g_dest = graph.all_edges()
    s2d_loop = g_src - g_dest
    src, dest = g_src[s2d_loop != 0], g_dest[s2d_loop != 0]
    graph_with_out_loop = DGLGraph()
    graph_with_out_loop.add_nodes(graph.number_of_nodes())
    graph_with_out_loop.add_edges(src, dest)
    for key, value in graph.ndata.items():
        graph_with_out_loop.ndata[key] = value
    return graph_with_out_loop

def reorginize_self_loop_edges(graph: DGLGraph):
    g_src, g_dest = graph.all_edges()
    s2d_loop = g_src - g_dest
    src, dest = g_src[s2d_loop != 0], g_dest[s2d_loop != 0]
    graph_reorg = DGLGraph()
    graph_reorg.add_nodes(graph.number_of_nodes())
    graph_reorg.add_edges(src, dest)
    self_loop_edge_number = (s2d_loop ==0).sum().item()
    if self_loop_edge_number > 0:
        self_src, self_dest = g_src[s2d_loop == 0], g_dest[s2d_loop == 0]
        graph_reorg.add_edges(self_src, self_dest)
    for key, value in graph.ndata.items():
        graph_reorg.ndata[key] = value
    return graph_reorg, self_loop_edge_number

def graph_to_undirected(graph: DGLGraph):
    g_src, g_dest = graph.all_edges()
    s2d_loop = g_src - g_dest
    src, dest = g_src[s2d_loop != 0], g_dest[s2d_loop != 0]
    self_loop_edge_number = s2d_loop == 0
    undirected_graph = DGLGraph()
    undirected_graph.add_nodes(graph.number_of_nodes())
    undirected_graph.add_edges(src, dest)
    undirected_graph.add_edges(dest, src)
    if self_loop_edge_number.sum() > 0:
        self_src, self_dest = g_src[s2d_loop == 0], g_dest[s2d_loop == 0]
        undirected_graph.add_edges(self_src, self_dest)
    for key, value in graph.ndata.items():
        undirected_graph.ndata[key] = value
    return undirected_graph