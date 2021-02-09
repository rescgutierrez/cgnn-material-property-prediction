import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.pytorch import GraphConv
import dgl

class CG_CNN_Layer(nn.Module):
    def __init__(self, in_feats):
        super(CG_CNN_Layer, self).__init__()
        self.linearf = nn.Linear(2 * in_feats + 10, in_feats)
        self.linears = nn.Linear(2 * in_feats + 10, in_feats)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        self.linearf.reset_parameters()
        self.linears.reset_parameters()

    def CGCNN_message(self, edges):
        msg = th.cat((edges.src['env'], edges.dst['env'],
                      edges.data['dist']), dim=-1)
        msg = (th.sigmoid(self.linearf(msg))) * (F.softplus(self.linears(msg)))
        return {'m': msg}

    def forward(self, g, feature):
        with g.local_scope():
            g.ndata['env'] = feature
            g.update_all(message_func=self.CGCNN_message,
                         reduce_func=fn.sum(msg='m',out='m_sum'))
            env = g.ndata['env'] + g.ndata['m_sum']
            return env


class Net(nn.Module):

    def __init__(self, in_feats, n_conv, neuron_ratios, activation):
        super(Net, self).__init__()
        self.conv1 = CG_CNN_Layer(in_feats)
        if n_conv>1:
            self.conv2 = CG_CNN_Layer(in_feats)
            if n_conv>2:
                self.conv3 = CG_CNN_Layer(in_feats)
        self.n_conv = n_conv
        self.mlp21 = nn.Linear(in_feats, neuron_ratios[1][0] * in_feats)
        self.mlp22 = nn.Linear(neuron_ratios[1][0] * in_feats, neuron_ratios[1][1] * in_feats)
        self.mlp23 = nn.Linear(neuron_ratios[1][1] * in_feats, 1)
        self.activation = activation()
        self.pooling = dgl.nn.pytorch.glob.AvgPooling()
        

    def forward(self, graphs):
        
        graphs = dgl.add_self_loop(graphs)
        out = self.conv1(graphs, graphs.ndata['Z'])
        if self.n_conv>1:
            out = self.conv2(graphs, out)
            if self.n_conv>2:
                out = self.conv3(graphs, out)
        #print(out.shape)
        with graphs.local_scope():
            graphs.ndata['env'] = out
            out = self.pooling(graphs,graphs.ndata['env'])
            out = self.activation(self.mlp21(out))
            out = self.activation(self.mlp22(out))
            out = self.mlp23(out)

            return out