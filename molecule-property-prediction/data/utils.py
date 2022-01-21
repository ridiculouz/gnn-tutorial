# from .encode import MolGraph
import dgl
from torch.utils.data import Dataset, DataLoader
import torch
from typing import List

import csv
import numpy as np
import pandas as pd
from data.encode import MolGraph, get_graph_from_smiles
import copy

class ChemDataset(Dataset):
    def __init__(self, graphs, afeats, bfeats, properties=None):
        super(ChemDataset, self).__init__()
        # dgl graph, atom feature, bond feature, label
        self.graphs = graphs
        self.afeats = afeats
        self.bfeats = bfeats
        self.labels = properties

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, index):
        if index >= len(self.graphs):
            raise ValueError("Out of index when getting example from dataset.")
        return self.graphs[index], self.afeats[index], self.bfeats[index], self.labels[index]

def chem_regression_collate_fn(batch):
    graphs, afeats, bfeats, labels = [], [], [], []
    for d in batch:
        graphs.append(d[0])
        afeats.append(torch.FloatTensor(d[1]))
        bfeats.append(torch.FloatTensor(d[2]))
        labels.append(d[3])
    graphs = dgl.batch(graphs)
    afeats = torch.cat(afeats, dim=0)
    bfeats = torch.cat(bfeats, dim=0)
    labels = torch.FloatTensor(labels)
    return graphs, afeats, bfeats, labels

def chem_classification_collate_fn(batch):
    graphs, afeats, bfeats, labels = [], [], [], []
    for d in batch:
        graphs.append(d[0])
        afeats.append(torch.FloatTensor(d[1]))
        bfeats.append(torch.FloatTensor(d[2]))
        d[3][np.isnan(d[3])] = -1
        labels.append(d[3])
    graphs = dgl.batch(graphs)
    afeats = torch.cat(afeats, dim=0)
    bfeats = torch.cat(bfeats, dim=0)
    labels = torch.LongTensor(labels)
    return graphs, afeats, bfeats, labels

def load_data(data_name, set_name='train', max_num=-1):
    # return a dataset
    if set_name == 'train':
        file_name = f'data/csvs/{data_name}-train.csv'
    elif set_name == 'test':
        file_name = f'data/csvs/{data_name}-test.csv'
    elif set_name == 'eval':
        file_name = f'data/csvs/{data_name}-eval.csv'
    else:
        assert False

    df = pd.read_csv(file_name)
    values: np.ndarray = df.values
    list_smiles = values[:, 0].astype(np.str)
    properties = values[:, 1:].astype(np.float)
    graphs = [get_graph_from_smiles(smiles) for smiles in list_smiles]
    dglgraphs = []
    afeats = []
    bfeats = []
    # adim = graphs[0].atom_features.shape[1]
    # bdim = graphs[0].bond_features.shape[1]
    discard_indices = []

    for idx, molg in enumerate(graphs):
        if molg == None:
            discard_indices.append(idx)
            continue
        num_nodes = molg.atom_features.shape[0]

        # duplicate_index = []
        # 
        # for i in range(len(molg.start_indices)):
        #     for j in range(i+1, len(molg.start_indices)):
        #         if molg.start_indices[i] == molg.end_indices[j] and molg.end_indices[i] == molg.start_indices[j]:
        #             duplicate_index.append(j)
        #             print("DUPLICATE")
        #         if molg.start_indices[i] == molg.start_indices[j] and molg.end_indices[i] == molg.end_indices[j]:
        #             duplicate_index.append(j)
        #             print("DUPLICATE")
        #         if molg.start_indices[i] == molg.end_indices[i]:
        #             duplicate_index.append(i)
        #             print("DUPLICATE")

        u = np.concatenate((molg.start_indices, molg.end_indices), axis=0)
        v = np.concatenate((molg.end_indices, molg.start_indices), axis=0)
        double_bfeat = np.concatenate((molg.bond_features, copy.deepcopy(molg.bond_features)), axis=0)
        dglg = dgl.graph((u, v), num_nodes=num_nodes)
        assert len(dglg.nodes()) == num_nodes
        dglgraphs.append(dglg)
        afeats.append(molg.atom_features)
        bfeats.append(double_bfeat)

    mask = np.ones(len(properties), dtype=bool)
    mask[discard_indices] = False
    properties = properties[mask]

    assert len(properties) == len(dglgraphs)
    
    return ChemDataset(dglgraphs, afeats, bfeats, properties)
