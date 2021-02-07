"""Pytorch dataset.

Single sample load of atom geometry (xyz file) into ase.atoms.atoms
object, and conversion to graph
"""
from ase.io import read
import ase
import torch
from torch.utils.data import Dataset
import numpy as np
from ase import neighborlist as asn
import csv
import torch as th
from dgl import DGLGraph
from sklearn.model_selection import ShuffleSplit


class CrystalsDataset(Dataset):
    """Crystals dataset."""

    def __init__(self, idxs, root_dir=""):
        """.

        Args:
            root_dir (string): project directory with data folder.
            target (tuple): tuple with target property values.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            idxs (iterable): to allow different dataset splits
        """
        self.root_dir = root_dir
        self.idxs = np.array(idxs)

        self.target = []
        with open(f'{self.root_dir}train.csv', 'r') as tf:
            next(tf)
            reader = csv.reader(tf)
            for row in reader:
                self.target.append(row[-1])
        del(tf, reader, row)
        self.target = np.array(tuple(self.target), dtype=np.float)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):

        def transform(system: ase.atoms.Atoms):
            """Create graph from atoms object."""

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

            # Get interaction sphere radius for each atom
            r_cuts = asn.natural_cutoffs(system)

            # Get three lists with atom-neighbor-distance relations
            nl = asn.neighbor_list('ijd', system, r_cuts)

            # Atomic numbers
            Zs = system.numbers

            # The node labels are vectors with one hot encoded atomic numbers
            nodes_atts = [Zs_encode[Z]
                          for Z in Zs]
            nodes_atts = th.Tensor(nodes_atts).type(th.int)

            # The edge attributes are distances one hot encoded according to bins in
            # function dist_value_to_onehot
            edges_index = [th.Tensor(nl[0]).type(
                th.int), th.Tensor(nl[1]).type(th.int)]
            edges_atts = th.Tensor(dist_value_to_onehot(nl[2])).type(th.FloatTensor)
            
            g = DGLGraph()
            g.add_nodes(nodes_atts.shape[0])

            g.ndata['Z'] = nodes_atts

            g.add_edges(edges_index[0], edges_index[1])
            g.edata['dist'] = edges_atts

            return g

        if torch.is_tensor(idx):
            idx = idx.tolist()

        target = (self.target[self.idxs[idx]])
        system = read(
            f"train/{self.idxs[idx]+1}/aseCL_geometry.xyz")

        system = transform(system)

        # sample = {'system': system, 'target': target}

        return system, target, self.idxs[idx]

# Dataset train, valid, test partitions
def partitions():
    """
    Returns:
    (train_idxs, valid_idxs, test_idxs)
    
    """
    idxs = np.arange(0, 2400, 1)
    spl1 = ShuffleSplit(
        n_splits=1, test_size=0.20, random_state=0).split(idxs)
    spl1 = tuple(spl1)
    train_idxs, valid_idxs = spl1[0][0], spl1[0][1]

    spl2 = ShuffleSplit(
        n_splits=1, test_size=1. / 7., random_state=0).split(train_idxs)
    spl2 = tuple(spl2)
    train_idxs, test_idxs = spl2[0][0], spl2[0][1]
    del(idxs)
    return train_idxs, valid_idxs, test_idxs