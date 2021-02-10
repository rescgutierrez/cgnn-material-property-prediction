"""Pytorch dataset.

Single sample load of atom geometry (xyz file) into ase.atoms.atoms
object, and conversion to graph
"""
import torch
from torch.utils.data import Dataset
import torch as th
from dgl import DGLGraph
from sklearn.model_selection import ShuffleSplit
import dgl


class GraphDataset(Dataset):
    """Crystals dataset."""

    def __init__(self, filename, idxs):
        """.

        """
        self.filename = filename
        self.idxs = idxs

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
            
        i = int(self.idxs[idx])
        graphs, labels = dgl.data.load_graphs(self.filename,[i-1])
        

        return graphs, labels, self.idxs[idx]
