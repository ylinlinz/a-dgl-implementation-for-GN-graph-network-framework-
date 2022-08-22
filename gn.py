import torch
import torch.nn as nn
import torch.nn.functional as f

import dgl
from dgl import function as fn
from dgl.utils import expand_as_pair

import numpy as np


def reset_parameters(seq):
    gain = nn.init.calculate_gain('relu')
    for net in seq:
        if isinstance(net, nn.Linear):
            nn.init.xavier_normal_(net.weight, gain=gain)
            nn.init.zeros_(net.bias)


class EdgeBlock(nn.Module):
    def __init__(self, graph_feat_size, node_feat_size, edge_feat_size):
        super(EdgeBlock, self).__init__()
        self.f_e = nn.Sequential(
            nn.Linear(graph_feat_size + 2 * node_feat_size + edge_feat_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, edge_feat_size),
        )
        reset_parameters(self.f_e)

    def forward(self, g, ns, nr, e, ei):
        eio = f.one_hot(ei)
        t_num = torch.sum(eio, dim=0).unsqueeze(1)

        x = torch.cat([g, ns, nr, e], dim=-1)
        a_feats = self.f_e(x).unsqueeze(1).repeat([1, eio.shape[1], 1])

        t_feats = torch.div(torch.sum(a_feats*eio.unsqueeze(2), dim=0), t_num)

        return torch.index_select(t_feats, 0, ei)


class NodeBlock(nn.Module):
    def __init__(self, graph_feat_size, node_feat_size, edge_feat_size):
        super(NodeBlock, self).__init__()
        self.f_n = nn.Sequential(
            nn.Linear(graph_feat_size + node_feat_size + edge_feat_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, node_feat_size),
        )
        reset_parameters(self.f_n)

    def forward(self, g, n, e):
        x = torch.cat([g, n, e], dim=-1)
        return self.f_n(x)


class GraphBlock(nn.Module):
    def __init__(self, graph_feat_size, node_feat_size, edge_feat_size):
        super(GraphBlock, self).__init__()
        self.f_g = nn.Sequential(
            nn.Linear(graph_feat_size + node_feat_size + edge_feat_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, graph_feat_size),
        )
        reset_parameters(self.f_g)

    def forward(self, g, n, e):
        x = torch.cat([g, n, e], dim=-1)
        return self.f_g(x)


class GNBlock(nn.Module):
    def __init__(self, ns, es, gs):
        super(GNBlock, self).__init__()
        self.n_block = NodeBlock(gs, ns, es)
        self.e_block = EdgeBlock(gs, ns, es)
        self.g_block = GraphBlock(gs, ns, es)

    def forward(self, graph, feats, efeats, ei, gfeat):
        assert isinstance(graph, dgl.DGLGraph)
        with graph.local_scope():
            feat_src, feat_dst = expand_as_pair(feats, graph)
            # 'h': current features (edge or node), 'hn': next edge features, 'nn': next node features, gn_feat: next global feature
            graph.srcdata['h'] = feat_src
            graph.edata['h'] = efeats
            graph.srcdata['g'] = gfeat.unsqueeze(0).repeat([feat_src.size()[0], 1])
            graph.edata['g'] = gfeat.unsqueeze(0).repeat([efeats.size()[0], 1])
            graph.apply_edges(lambda e: {'hn': self.e_block(e.data['g'], e.src['h'], e.dst['h'], e.data['h'], ei)})
            graph.update_all(fn.copy_e('hn', 'm'), fn.sum('m', 'ef'))
            graph.apply_nodes(lambda n: {'nn': self.n_block(n.data['g'], n.data['h'], n.data['ef'])})
            gn_feat = self.g_block(gfeat, torch.sum(graph.ndata['nn'], dim=0), torch.sum(graph.edata['hn'], dim=0))
            return graph.ndata['nn'], graph.edata['hn'], gn_feat


if __name__ == '__main__':
    
    # dgl graph
    g = dgl.graph(([0, 0, 1, 2], [1, 3, 2, 3]))
    
    # node features
    feat = torch.from_numpy(np.array([[3, 2], [2, 0], [1, 3], [0, 1]], dtype=np.float32))
    # edge features
    efeat = torch.from_numpy(np.array([[1, 3], [2, 1], [5, 7], [2, 1]], dtype=np.float32))
    # edge types
    ei = torch.from_numpy(np.array([1, 0, 2, 0]))
    # global featrue (or graph feature)
    gfeat = torch.from_numpy(np.array([2, 3, 8, 1], dtype=np.float32))

    net1 = GNBlock(2, 2, 4)
    net2 = GNBlock(2, 2, 4)
    h1 = net1(g, feat, efeat, ei, gfeat)
    h2 = net2(g, h1[0], h1[1], ei, h1[2])
