from dgl.function.message import copy_edge
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple

import dgl
import dgl.function as fn

class CMPNN(nn.Module):
    def __init__(self, atom_dim: int, bond_dim: int, config: Dict[str, Any]):
        super(CMPNN, self).__init__()
        self.atom_dim = atom_dim
        self.bond_dim = bond_dim
        self.undirected: bool = True
        self.output_dim: int = config['OUTPUT_DIM']
        self.hidden_dim: int = config['HIDDEN_DIM']
        self.depth: int = config['DEPTH']
        self.bias: bool = config['BIAS']
        self.drouput: float = config['DROPOUT']

        self.atom_embed = nn.Linear(self.atom_dim, self.hidden_dim)
        self.bond_embed = nn.Linear(self.bond_dim, self.hidden_dim)
        self.relu = nn.ReLU()
        self.dropout_layer = nn.Dropout(self.drouput)
        # self.output_embed = nn.Linear(self.hidden_dim, self.output_dim)
        self.W_bond = nn.Linear(self.hidden_dim, self.hidden_dim, bias=self.bias)
        self.W_atom = nn.Linear(3*self.hidden_dim, self.output_dim, bias=self.bias)
        self.readout_layer = nn.Linear(self.output_dim, self.output_dim, bias=self.bias)

        for depth in range(self.depth):
            self._modules[f'W_n_{depth}'] = nn.Linear(2*self.hidden_dim, self.hidden_dim, bias=self.bias)
            

    def forward(self, atom_features: torch.Tensor, bond_features: torch.Tensor,
                g: dgl.DGLGraph) -> torch.Tensor:
        #
        # g: a batch of graph
        # return a batch of graph representation
        #
        input_atom = self.relu(self.atom_embed(atom_features))
        input_bond = self.relu(self.bond_embed(bond_features))

        intermediate_atom = input_atom.clone()
        intermediate_bond = input_bond
        # message passing
        for depth in range(self.depth):
            # for v in V
            def reduce_fn(nodes):
                return {'m': torch.sum(nodes.mailbox['mi'], dim=1)*torch.max(nodes.mailbox['mi'], dim=1)[0]}
            g.edata['h'] = intermediate_bond
            g.update_all(copy_edge('h', 'mi'), reduce_fn)
            message_atom = g.ndata['m']
            # print(message_atom.size())
            # print(intermediate_atom.size())
            # print(input_atom.size())
            # print(len(g.nodes()))
            intermediate_atom = self._modules[f'W_n_{depth}'](torch.cat([message_atom, intermediate_atom], dim=1))
            intermediate_atom = self.dropout_layer(self.relu(intermediate_atom))
            g.ndata['h'] = intermediate_atom
            # for e in E
            if depth == self.depth - 1:
                continue
            srcs, dsts = g.edges()
            reversed_intermediate_bond = g.edata['h'][g.edge_ids(dsts, srcs)]
            message_bond = g.ndata['h'][srcs] - reversed_intermediate_bond
            intermediate_bond = input_bond + self.W_bond(message_bond)
            intermediate_bond = self.dropout_layer(self.relu(intermediate_bond))
        output_atom = self.W_atom(torch.cat([input_atom, intermediate_atom, message_atom], dim=1))
        output_atom = self.dropout_layer(self.relu(output_atom))
        # readout
        # sum atom feature and mlp
        g.ndata['rep'] = output_atom
        mol_graphs = dgl.unbatch(g)
        graph_rep_lists = []
        for mol_g in mol_graphs:
            grep = self.readout_layer(torch.sum(mol_g.ndata['rep'], dim=0))
            grep = self.dropout_layer(self.relu(grep))
            graph_rep_lists.append(grep)
        graph_rep = torch.stack(graph_rep_lists, dim=0)
        
        return graph_rep # [batch_size, hidden]


class MoleculeNet(nn.Module):
    def __init__(self, atom_dim: int, bond_dim: int, config: Dict[str, Any]):
        super(MoleculeNet, self).__init__()
        self.config = config
        self.atom_dim = atom_dim
        self.bond_dim = bond_dim
        self.bias: bool = config['BIAS']
        self.output_dim: int = config['OUTPUT_DIM']
        self.drouput: float = config['DROPOUT']
        self.task = config["TASK"]                          # regression or classification
        if self.task == "classification":
            self.head: Tuple[int, int] = config["HEAD"]     # task number, class number
            self.MLP = nn.Linear(self.output_dim, self.head[0]*self.head[1], bias=self.bias)
        elif self.task == "regression":
            self.MLP = nn.Linear(self.output_dim, 1, bias=self.bias)
        else:
            raise ValueError("Unknown task type.")

        self.MPNEncoder = CMPNN(atom_dim, bond_dim, config)
        # self.relu = nn.ReLU()
        # self.dropout_layer = nn.Dropout(self.drouput)

    def forward(self, atom_features, bond_features, g):
        graph_rep = self.MPNEncoder(atom_features, bond_features, g)

        return self.MLP(graph_rep)   
  



def index_select_ND(source: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """
    Selects the message features from source corresponding to the atom or bond indices in index.
    :param source: A tensor of shape (num_bonds, hidden_size) containing message features.
    :param index: A tensor of shape (num_atoms/num_bonds, max_num_bonds) containing the atom or bond
    indices to select from source.
    :return: A tensor of shape (num_atoms/num_bonds, max_num_bonds, hidden_size) containing the message
    features corresponding to the atoms/bonds specified in index.
    """
    index_size = index.size()  # (num_atoms/num_bonds, max_num_bonds)
    suffix_dim = source.size()[1:]  # (hidden_size,)
    final_size = index_size + suffix_dim  # (num_atoms/num_bonds, max_num_bonds, hidden_size)

    target = source.index_select(dim=0, index=index.view(-1))  # (num_atoms/num_bonds * max_num_bonds, hidden_size)
    target = target.view(final_size)  # (num_atoms/num_bonds, max_num_bonds, hidden_size)
    
    return target