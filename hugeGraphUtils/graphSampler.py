from dgl import DGLGraph
import torch
from time import time
def graph_k_neighbor_sampler(graph: DGLGraph, knn: int):
    start_time = time()
    graph = graph.local_var()  # the graph should be added a self-loop edge
    num_nodes = graph.number_of_nodes()
    nids = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
    graph.ndata.update({'node_id': nids})
    def send_edge_message(edges):
        return {'m_src_id': edges.src['node_id']}

    def knn_sampler_reduce_func(nodes):
        node_ids = nodes.mailbox['m_src_id']
        node_ids = node_ids.squeeze(dim=-1)
        sample_num, neighbor_num = node_ids.shape[0], node_ids.shape[1]
        sample_knns = -torch.ones((node_ids.shape[0], knn), dtype=torch.long)
        if neighbor_num <= knn:
            sample_knns[:, :neighbor_num] = node_ids
        else:
            for i in range(sample_num):
                node_ids[i] = node_ids[i][torch.randperm(neighbor_num)]
            sample_knns[:, :knn] = node_ids[:, :knn]
        return {'knn_ids': sample_knns}

    graph.register_reduce_func(knn_sampler_reduce_func)
    graph.register_message_func(send_edge_message)
    graph.update_all(message_func=send_edge_message, reduce_func=knn_sampler_reduce_func)

    knnids = graph.ndata.pop('knn_ids').unsqueeze(dim=-1)
    kcopyids = torch.repeat_interleave(nids, repeats=knn, dim=-1).unsqueeze(dim=-1)
    edge_pairs = torch.cat([knnids, kcopyids], dim=-1)
    edge_pairs = edge_pairs.view(graph.number_of_nodes() * knn, -1)
    # print(edge_pairs.shape)
    edge_pairs = edge_pairs[edge_pairs[:,0]>=0, :]
    print(edge_pairs.shape)
    # print(nids.shape, knnids.shape, edge_pairs.shape, kcopyids.shape)
    new_graph = graph_construction(graph=graph, edge_pairs=edge_pairs)
    print('Sampling takes {}'.format(time() - start_time))
    return new_graph


def graph_ratio_neighbor_sampler(graph: DGLGraph, sample_ratio: float=0.1):
    start_time = time()
    graph = graph.local_var()  # the graph should be added a self-loop edge
    num_nodes = graph.number_of_nodes()
    nids = torch.arange(0, num_nodes, dtype=torch.long)
    indegrees = graph.in_degrees(nids)
    max_indegree = indegrees.max().item()
    knn = int(max_indegree * sample_ratio) + 1
    # print(max_indegree, knn)
    nids = nids.view(-1, 1)
    graph.ndata.update({'node_id': nids})
    def send_edge_message(edges):
        return {'m_src_id': edges.src['node_id']}

    def knn_sampler_reduce_func(nodes):
        node_ids = nodes.mailbox['m_src_id']
        node_ids = node_ids.squeeze(dim=-1)
        sample_num, neighbor_num = node_ids.shape[0], node_ids.shape[1]
        sample_knns = -torch.ones((node_ids.shape[0], knn), dtype=torch.long).to(node_ids.device)
        if neighbor_num <= knn:
            sample_knns[:, :neighbor_num] = node_ids
        else:
            for i in range(sample_num):
                node_ids[i] = node_ids[i][torch.randperm(neighbor_num)]
            sample_knns[:, :knn] = node_ids[:, :knn]
        return {'knn_ids': sample_knns}

    graph.register_reduce_func(knn_sampler_reduce_func)
    graph.register_message_func(send_edge_message)
    graph.update_all(message_func=send_edge_message, reduce_func=knn_sampler_reduce_func)

    knnids = graph.ndata.pop('knn_ids').unsqueeze(dim=-1)
    kcopyids = torch.repeat_interleave(nids, repeats=knn, dim=-1).unsqueeze(dim=-1)
    edge_pairs = torch.cat([knnids, kcopyids], dim=-1)
    edge_pairs = edge_pairs.view(graph.number_of_nodes() * knn, -1)
    edge_pairs = edge_pairs[edge_pairs[:, 0] >= 0, :]
    new_graph = graph_construction(graph=graph, edge_pairs=edge_pairs)
    print('Sampling takes {}'.format(time() - start_time))
    return new_graph

def graph_construction(graph: DGLGraph, edge_pairs):
    graph = graph.local_var()
    new_graph = DGLGraph()
    num_nodes = graph.number_of_nodes()
    new_graph.add_nodes(num_nodes)
    src_nodes, dest_nodes = edge_pairs[:,0], edge_pairs[:,1]
    edge_ids = graph.edge_ids(src_nodes, dest_nodes)
    # print('here', edge_ids.shape, src_nodes.shape)
    new_graph.add_edges(src_nodes, dest_nodes)
    for key, value in graph.ndata.items():
        new_graph.ndata[key] = value
    for key, value in graph.edata.items():
        new_graph.edata[key] = value[edge_ids]
    return new_graph