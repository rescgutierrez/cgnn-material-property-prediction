#!/usr/bin/env python
# coding: utf-8

# In[1]:


from dgl.data.utils import save_graphs
from ase.io import read
import ase
from ase import neighborlist as asn
import numpy as np
import csv
import torch as th
from dgl import DGLGraph
import dgl
import torch as th


# In[2]:


graph_labels = []
with open(f'train.csv', 'r') as tf:
    next(tf)
    reader = csv.reader(tf)
    for row in reader:
        graph_labels.append(float(row[-1]))
del(tf, reader, row)


# In[3]:


graphs = []

def dist_value_to_onehot(dist):
    bins = np.array([0.7, 1.15, 1.6, 2.05, 2.5, 2.95, 3.40, 3.85, 4.3,
                     4.75, 5.2],
                    dtype=np.float)
    value_pos = np.searchsorted(bins, dist)
    return np.eye(10, dtype=np.int)[value_pos - 1]


Zs_encode = {
    13 : [1,0,0,0],

    8 : [0,1,0,0],

    49 : [0,0,1,0],

    31 : [0,0,0,1],
}

for i in range(2400):
    
    system = read(f"train/{i+1}/aseCL_geometry.xyz")
    
    # Get interaction sphere radius for each atom
    r_cuts = asn.natural_cutoffs(system)

    # Get three lists with atom-neighbor-distance relations
    nl = asn.neighbor_list('ijd', system, r_cuts)

    # Atomic numbers
    Zs = system.numbers

    # The node labels are vectors with one hot encoded atomic numbers
    nodes_atts = [Zs_encode[Z]
                  for Z in Zs]
    nodes_atts = th.Tensor(nodes_atts).type(th.float)

    # The edge attributes are distances one hot encoded according to bins in
    # function dist_value_to_onehot
    edges_index = (th.Tensor(nl[0]).type(
        th.int), th.Tensor(nl[1]).type(th.int))
    edges_atts = th.Tensor(dist_value_to_onehot(nl[2])).type(th.float)
    
    g = dgl.graph(edges_index, num_nodes=nodes_atts.shape[0])
    g.ndata['Z'] = nodes_atts
    g.edata['dist'] = edges_atts
    
    graphs.append(g)


# In[4]:


graph_labels = {"glabel": th.tensor(graph_labels, dtype = th.float)}
save_graphs("./data.bin", graphs, graph_labels)

