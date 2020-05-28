import torch
import torch.nn as nn
from dgl import DGLGraph
import numpy as np
from gdtransformer.graphtransformer import gtransformerlayer
from hugeGraphUtils.graphSampler import graph_k_neighbor_sampler, graph_ratio_neighbor_sampler

class GDTSampler(nn.Module):
    def __init__(self,
                 g: DGLGraph,
                 num_layers: int,
                 input_dim: int,
                 hidden_dim: int,
                 hop_num: int,
                 alpha: float,
                 num_classes: int,
                 heads: list,
                 top_k:int,
                 feat_drop: float,
                 attn_drop: float,
                 negative_slope: float,
                 edge_drop: float,
                 topk_type: str,
                 number_self_loops: int,
                 knn_sampler=5,
                 ratio_sampler=0.1,
                 self_loop=True,
                 undirected_graph=True,
                 project_dim=-1):
        super(GDTSampler, self).__init__()
        self.g = g
        self.gdt_layers = nn.ModuleList()
        self.self_loop = self_loop
        self.undirected_graph = undirected_graph
        self.number_self_loops = number_self_loops
        self.knn_sampler=knn_sampler
        self.ratio_sampler =ratio_sampler
        if project_dim > 1:
            self.project = nn.Linear(in_features=input_dim, out_features=project_dim)
            self.input_features = project_dim
        else:
            self.register_buffer('project', None)
            self.input_features = input_dim

        self.num_layers = num_layers
        self.edge_drop = edge_drop
        self.gdt_layers.append(gtransformerlayer(in_feats=self.input_features, hop_num=hop_num, top_k=top_k, num_heads=heads[0], hidden_dim=hidden_dim,
                                                 topk_type=topk_type,
                                                 alpha=alpha, negative_slope=negative_slope, feat_drop=feat_drop, attn_drop=attn_drop))
        for l in range(1, self.num_layers):
            self.gdt_layers.append(gtransformerlayer(in_feats=hidden_dim, hop_num=hop_num, hidden_dim=hidden_dim, num_heads=heads[l], top_k=top_k,
                                                     topk_type=topk_type, alpha=alpha, negative_slope=negative_slope, feat_drop=feat_drop, attn_drop=attn_drop))
        self.classifier = nn.Linear(in_features=hidden_dim, out_features=num_classes)
        self.feat_drop_out = nn.Dropout(p=feat_drop)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        if isinstance(self.classifier, nn.Linear):
            nn.init.xavier_normal_(self.classifier.weight.data)
        if self.project is not None and isinstance(self.project, nn.Linear):
            nn.init.xavier_normal_(self.project.weight.data)

    def forward(self, inputs):
        if self.project is not None:
            h = self.project(self.feat_drop_out(inputs))
        else:
            h = inputs
        # if self.training:
        #     g = self.graph_sampler()
        # else:
        #     g = self.g

        for l in range(self.num_layers):
            h, _ = self.gdt_layers[l](self.g, h)
        logits = self.classifier(h)
        return logits

    def graph_sampler(self):
        if self.knn_sampler > 0:
            new_g = graph_k_neighbor_sampler(graph=self.g, knn=self.knn_sampler)
        else:
            new_g = graph_ratio_neighbor_sampler(graph=self.g, sample_ratio=self.ratio_sampler)
        return new_g

    def layer_attention_node_features(self, inputs):
        number_edges = self.g.number_of_edges()
        layer_node_features, layer_attentions = [], []
        if self.project is not None:
            h = self.project(self.feat_drop_out(inputs))
        else:
            h = inputs
        for l in range(self.num_layers):
            drop_edge_ids = self.get_drop_edge_ids(number_edges)
            h, attentions = self.gdt_layers[l](self.g, h, drop_edge_ids)
            layer_node_features.append(h)
            layer_attentions.append(attentions)
        logits = self.classifier(h)
        return logits, layer_node_features, layer_attentions