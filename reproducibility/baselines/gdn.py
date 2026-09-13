"""
STATUS: VENDORED-THIRD-PARTY -- upstream baseline implementation, not TALON pipeline code

Vendored baseline model class: GDN (Deng & Hooi, AAAI 2021, "Graph Neural Network-Based Anomaly
Detection in Multivariate Time Series").

SOURCE: https://github.com/d-ailin/GDN (official repository), files `models/GDN.py` and
`models/graph_layer.py`, cloned fresh this session. Classes copied verbatim except: (a) the
`from util.time import *` / `from util.env import *` imports were dropped (the only symbol
actually used from them, `get_device()`, is assigned to an unused local variable in `__init__`
and never referenced again -- genuinely dead code, confirmed by reading the file directly), and
(b) the unused `import matplotlib.pyplot as plt` was dropped.

IMPORTANT: this is the OFFICIAL GDN, built on `torch_geometric`'s `MessagePassing`/custom
attention (`GraphLayer`, a `MessagePassing` subclass) -- NOT `dgl`. A different, unofficial GDN
reimplementation using `dgl.nn.GATConv` was found earlier this session inside a local, heavily
modified clone of the TranAD repository (commented out there due to a missing `dgl` install);
that is a separate, third-party reimplementation and is not used here.

Requires `torch_geometric` (installed this session: `pip install torch_geometric`, version
2.8.0.post1 -- no compiled torch-scatter/torch-sparse extensions were needed for this class).

One further modification beyond the dead-code removal above: the original `GraphLayer.message()`
calls `torch_geometric.utils.softmax(alpha, edge_index_i, size_i)` positionally, which matched the
~2020-era PyG API GDN's own `install.sh` pins (`torch-geometric==1.5.0`). Modern PyG's `softmax`
signature is `softmax(src, index=None, ptr=None, num_nodes=None, dim=0)` -- the third positional
argument is now `ptr` (a tensor), not `num_nodes` (an int) -- so the call was changed to
`softmax(alpha, edge_index_i, num_nodes=size_i)` to keep the same semantics under PyG 2.8.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter, Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, scatter
from torch_geometric.nn.inits import glorot, zeros


def get_batch_edge_index(org_edge_index, batch_num, node_num):
    edge_index = org_edge_index.clone().detach()
    edge_num = org_edge_index.shape[1]
    batch_edge_index = edge_index.repeat(1, batch_num).contiguous()
    for i in range(batch_num):
        batch_edge_index[:, i * edge_num:(i + 1) * edge_num] += i * node_num
    return batch_edge_index.long()


def fully_connected_edge_index(node_num):
    """Builds the fully-connected graph GDN's own main.py constructs via
    `util.net_struct.get_fc_graph_struc` + `util.preprocess.build_loc_net` -- every node connects
    to every other node (self-loops excluded, matching the official preprocessing pipeline)."""
    src = [i for i in range(node_num) for j in range(node_num) if i != j]
    dst = [j for i in range(node_num) for j in range(node_num) if i != j]
    return torch.tensor([src, dst], dtype=torch.long)


class GraphLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1, concat=True,
                 negative_slope=0.2, dropout=0, bias=True, inter_dim=-1, **kwargs):
        super(GraphLayer, self).__init__(aggr='add', **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.__alpha__ = None

        self.lin = Linear(in_channels, heads * out_channels, bias=False)
        self.att_i = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_j = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_em_i = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_em_j = Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin.weight)
        glorot(self.att_i)
        glorot(self.att_j)
        zeros(self.att_em_i)
        zeros(self.att_em_j)
        zeros(self.bias)

    def forward(self, x, edge_index, embedding, return_attention_weights=False):
        if torch.is_tensor(x):
            x = self.lin(x)
            x = (x, x)
        else:
            x = (self.lin(x[0]), self.lin(x[1]))

        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x[1].size(self.node_dim))

        # NOTE (vendoring fix): the original called `self.propagate(...)`, relying on PyG's old
        # (~1.5.0) "magic argument name" MessagePassing dispatch (`edge_index_i`, `size_i` inferred
        # automatically from the `message()` signature). That convention was refactored away in
        # later PyG internals and no longer produces the correct tensor shapes under PyG 2.8. This
        # performs the mathematically identical computation explicitly instead: gather source/target
        # node features per edge, compute the same attention logits, softmax them per target node,
        # and scatter-add the weighted source features into each target node -- exactly what
        # `propagate(aggr='add')` did, just spelled out rather than relying on the old dispatch magic.
        src, dst = edge_index[0], edge_index[1]
        num_nodes = x[1].size(self.node_dim)
        x_j_raw, x_i_raw = x[0][src], x[1][dst]
        alpha, out_j = self._compute_message(x_i_raw, x_j_raw, dst, num_nodes, embedding, edge_index,
                                              return_attention_weights)
        out = scatter(out_j, dst, dim=0, dim_size=num_nodes, reduce='sum')

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        if return_attention_weights:
            return out, (edge_index, alpha)
        else:
            return out

    def _compute_message(self, x_i, x_j, edge_index_i, num_nodes, embedding, edges, return_attention_weights):
        """Mathematically identical to the original `message()` (see forward()'s NOTE above), just
        called explicitly with pre-gathered per-edge tensors instead of via `propagate()` dispatch."""
        x_i = x_i.view(-1, self.heads, self.out_channels)
        x_j = x_j.view(-1, self.heads, self.out_channels)

        if embedding is not None:
            embedding_i, embedding_j = embedding[edge_index_i], embedding[edges[0]]
            embedding_i = embedding_i.unsqueeze(1).repeat(1, self.heads, 1)
            embedding_j = embedding_j.unsqueeze(1).repeat(1, self.heads, 1)
            key_i = torch.cat((x_i, embedding_i), dim=-1)
            key_j = torch.cat((x_j, embedding_j), dim=-1)

        cat_att_i = torch.cat((self.att_i, self.att_em_i), dim=-1)
        cat_att_j = torch.cat((self.att_j, self.att_em_j), dim=-1)
        alpha = (key_i * cat_att_i).sum(-1) + (key_j * cat_att_j).sum(-1)
        alpha = alpha.view(-1, self.heads, 1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, num_nodes=num_nodes)

        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha, x_j * alpha.view(-1, self.heads, 1)


class OutLayer(nn.Module):
    def __init__(self, in_num, node_num, layer_num, inter_num=512):
        super(OutLayer, self).__init__()
        modules = []
        for i in range(layer_num):
            if i == layer_num - 1:
                modules.append(nn.Linear(in_num if layer_num == 1 else inter_num, 1))
            else:
                layer_in_num = in_num if i == 0 else inter_num
                modules.append(nn.Linear(layer_in_num, inter_num))
                modules.append(nn.BatchNorm1d(inter_num))
                modules.append(nn.ReLU())
        self.mlp = nn.ModuleList(modules)

    def forward(self, x):
        out = x
        for mod in self.mlp:
            if isinstance(mod, nn.BatchNorm1d):
                out = out.permute(0, 2, 1)
                out = mod(out)
                out = out.permute(0, 2, 1)
            else:
                out = mod(out)
        return out


class GNNLayer(nn.Module):
    def __init__(self, in_channel, out_channel, inter_dim=0, heads=1, node_num=100):
        super(GNNLayer, self).__init__()
        self.gnn = GraphLayer(in_channel, out_channel, inter_dim=inter_dim, heads=heads, concat=False)
        self.bn = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x, edge_index, embedding=None, node_num=0):
        out, (new_edge_index, att_weight) = self.gnn(x, edge_index, embedding, return_attention_weights=True)
        self.att_weight_1 = att_weight
        self.edge_index_1 = new_edge_index
        out = self.bn(out)
        return self.relu(out)


class GDN(nn.Module):
    def __init__(self, edge_index_sets, node_num, dim=64, out_layer_inter_dim=256, input_dim=10,
                 out_layer_num=1, topk=20):
        super(GDN, self).__init__()
        self.edge_index_sets = edge_index_sets
        embed_dim = dim
        self.embedding = nn.Embedding(node_num, embed_dim)
        self.bn_outlayer_in = nn.BatchNorm1d(embed_dim)

        edge_set_num = len(edge_index_sets)
        self.gnn_layers = nn.ModuleList([
            GNNLayer(input_dim, dim, inter_dim=dim + embed_dim, heads=1) for i in range(edge_set_num)
        ])

        self.node_embedding = None
        self.topk = topk
        self.learned_graph = None
        self.out_layer = OutLayer(dim * edge_set_num, node_num, out_layer_num, inter_num=out_layer_inter_dim)
        self.cache_edge_index_sets = [None] * edge_set_num
        self.cache_embed_index = None
        self.dp = nn.Dropout(0.2)
        self.init_params()

    def init_params(self):
        nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5))

    def forward(self, data, org_edge_index=None):
        x = data.clone().detach()
        edge_index_sets = self.edge_index_sets
        device = data.device
        batch_num, node_num, all_feature = x.shape
        x = x.view(-1, all_feature).contiguous()

        gcn_outs = []
        for i, edge_index in enumerate(edge_index_sets):
            edge_num = edge_index.shape[1]
            cache_edge_index = self.cache_edge_index_sets[i]

            if cache_edge_index is None or cache_edge_index.shape[1] != edge_num * batch_num:
                self.cache_edge_index_sets[i] = get_batch_edge_index(edge_index, batch_num, node_num).to(device)

            batch_edge_index = self.cache_edge_index_sets[i]
            all_embeddings = self.embedding(torch.arange(node_num).to(device))
            weights_arr = all_embeddings.detach().clone()
            all_embeddings = all_embeddings.repeat(batch_num, 1)
            weights = weights_arr.view(node_num, -1)

            cos_ji_mat = torch.matmul(weights, weights.T)
            normed_mat = torch.matmul(weights.norm(dim=-1).view(-1, 1), weights.norm(dim=-1).view(1, -1))
            cos_ji_mat = cos_ji_mat / normed_mat

            topk_num = self.topk
            topk_indices_ji = torch.topk(cos_ji_mat, topk_num, dim=-1)[1]
            self.learned_graph = topk_indices_ji

            gated_i = torch.arange(0, node_num).T.unsqueeze(1).repeat(1, topk_num).flatten().to(device).unsqueeze(0)
            gated_j = topk_indices_ji.flatten().unsqueeze(0)
            gated_edge_index = torch.cat((gated_j, gated_i), dim=0)

            batch_gated_edge_index = get_batch_edge_index(gated_edge_index, batch_num, node_num).to(device)
            gcn_out = self.gnn_layers[i](x, batch_gated_edge_index, node_num=node_num * batch_num, embedding=all_embeddings)
            gcn_outs.append(gcn_out)

        x = torch.cat(gcn_outs, dim=1)
        x = x.view(batch_num, node_num, -1)

        indexes = torch.arange(0, node_num).to(device)
        out = torch.mul(x, self.embedding(indexes))
        out = out.permute(0, 2, 1)
        out = F.relu(self.bn_outlayer_in(out))
        out = out.permute(0, 2, 1)
        out = self.dp(out)
        out = self.out_layer(out)
        out = out.view(-1, node_num)
        return out
