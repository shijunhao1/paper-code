import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, GINConv, SAGEConv, TransformerConv, global_add_pool


def build_conv(conv_type: str):
    if conv_type == "GCN":
        return GCNConv
    if conv_type == "GIN":
        return lambda i, h: GINConv(nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h)))
    if conv_type == "GAT":
        return GATConv
    if conv_type == "TransformerConv":
        return TransformerConv
    if conv_type == "SAGE":
        return SAGEConv
    raise KeyError("GNN type must be one of [GCN, GIN, GAT, TransformerConv, SAGE]")


class GNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layer=2, gnn_type="GCN", dropout=0.2):
        super().__init__()

        conv = build_conv(gnn_type)
        self.act = nn.LeakyReLU()
        self.dropout = dropout
        self.pool = global_add_pool
        self.output_dim = output_dim

        if n_layer < 1:
            raise ValueError(f"n_layer should >= 1 but got {n_layer}")
        if n_layer == 1:
            self.conv_layers = nn.ModuleList([conv(input_dim, output_dim)])
        elif n_layer == 2:
            self.conv_layers = nn.ModuleList([conv(input_dim, hidden_dim), conv(hidden_dim, output_dim)])
        else:
            layers = [conv(input_dim, hidden_dim)]
            for _ in range(n_layer - 2):
                layers.append(conv(hidden_dim, hidden_dim))
            layers.append(conv(hidden_dim, output_dim))
            self.conv_layers = nn.ModuleList(layers)

    def forward(self, x, edge_index, batch=None):
        for graph_conv in self.conv_layers[:-1]:
            x = graph_conv(x, edge_index)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        node_emb = self.conv_layers[-1](x, edge_index)

        if batch is None:
            return node_emb

        device = batch.device
        ones = torch.ones_like(batch).to(device)
        nodes_per_graph = global_add_pool(ones, batch)
        centric_idx = torch.cat((torch.LongTensor([0]).to(device), torch.cumsum(nodes_per_graph, dim=0)[:-1]))

        graph_emb = self.pool(node_emb, batch)
        return node_emb[centric_idx], graph_emb
