import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import time
from typing import Tuple, List
from functools import reduce
from sklearn.metrics import roc_auc_score

from torch.utils.data.dataloader import DataLoader

from data.utils import load_data, chem_regression_collate_fn, chem_classification_collate_fn
from data.encode import num_atom_features, num_bond_features
from model.CMPNN import MoleculeNet


config = {
    "OUTPUT_DIM": 128,
    "HIDDEN_DIM": 128,
    "DEPTH": 3,
    "BIAS": True,
    "DROPOUT": 0.2,
    "LR": 1e-3,
    "WEIGHT_DECAY": 5e-4,
    "LR_DECAY_STEP": 10,
    "LR_DECAY_RATE": 0.95, 
    "EPOCH": 500,
    "DATASET": "sars-split",
    "TASK": "classification",
    "HEAD": (13, 4),
    "SEED": 0,
    "DEVICE": "cuda:1",
}

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train(config):
    def evaluate_regression(loader, model):
        model.eval()
        with torch.no_grad():
            total_labels = []
            total_pred = []
            for (graphs, afeats, bfeats, labels) in loader: 
                graphs = graphs.to(config['DEVICE'])
                afeats = afeats.to(config['DEVICE'])
                bfeats = bfeats.to(config['DEVICE'])
                labels = labels.to(config['DEVICE'])
                pred = model(afeats, bfeats, graphs)
                total_labels.append(labels)
                total_pred.append(pred)
            total_labels = torch.cat(total_labels, dim=0)
            total_pred = torch.cat(total_pred, dim=0)
            rmse = torch.sqrt(F.mse_loss(total_pred, total_labels))
            rmse = float(rmse)
        print(f'\t\t\tRMSE: {rmse:.4f}')
        return rmse

    def evaluate_classification(loader, model):
        model.eval()
        with torch.no_grad():
            total_labels = []
            total_pred = []
            for (graphs, afeats, bfeats, labels) in loader: 
                graphs = graphs.to(config['DEVICE'])
                afeats = afeats.to(config['DEVICE'])
                bfeats = bfeats.to(config['DEVICE'])
                labels = labels.to(config['DEVICE'])
                pred = model(afeats, bfeats, graphs)
                labels = labels.view(-1)
                pred = pred.view(labels.size(0), -1)
                pred = F.softmax(pred, dim=-1)
                mask = labels != -1
                labels, pred = labels[mask], pred[mask]
                total_labels.append(labels)
                total_pred.append(pred)
            total_labels = torch.cat(total_labels, dim=0).cpu().numpy()
            total_pred = torch.cat(total_pred, dim=0).cpu().numpy()
            auc = roc_auc_score(total_labels, total_pred, multi_class="ovo")
        print(f'\t\t\tROC AUC score: {auc:.4f}')
        return auc
                
    # set random seed
    set_seed(config['SEED'])
    # load data
    print("\tLoading data...")
    train_dtset = load_data(config['DATASET'], set_name='train')
    eval_dtset = load_data(config['DATASET'], set_name='eval')
    print("\tLoading data from testing set...")
    test_dtset = load_data(config['DATASET'], set_name='test')
    if config['TASK'] == 'regression':
        custom_collate_fn = chem_regression_collate_fn
    elif config['TASK'] == 'classification':
        custom_collate_fn = chem_classification_collate_fn
    else:
        raise ValueError('Unknown config task.')
    train_dtloader = DataLoader(train_dtset, batch_size=64, shuffle=True, collate_fn=custom_collate_fn)
    eval_dtloader = DataLoader(eval_dtset, batch_size=128, shuffle=False, collate_fn=custom_collate_fn)
    test_dtloader = DataLoader(test_dtset, batch_size=128, shuffle=False, collate_fn=custom_collate_fn)
    # build model
    print('\tBuilding Model...')
    atom_dim = num_atom_features()
    bond_dim = num_bond_features()
    model = MoleculeNet(atom_dim, bond_dim, config)
    model.to(config['DEVICE'])
    print('\tStructure:')
    n_param = 0
    for name, param in model.named_parameters():
        print(f'\t\t{name}: {param.shape}')
        n_param += reduce(lambda x, y: x * y, param.shape)
    print(f'\t# Parameters: {n_param}')
    # build optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['LR'], weight_decay=config['WEIGHT_DECAY'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['LR_DECAY_STEP'], gamma=config['LR_DECAY_RATE'])
    # training
    if config['TASK'] == 'regression':
        best_metrics = 1e9
    elif config['TASK'] == 'classification':
        best_metrics = 0  
    for epoch in range(config['EPOCH']):
        print(f'##### IN EPOCH {epoch} #####')
        print('\tCurrent LR: {:.3e}'.format(optimizer.state_dict()['param_groups'][0]['lr']))
        print('\t\tTraining:')
        t0 = time.time()
        model.train()
        for (graphs, afeats, bfeats, labels) in train_dtloader:
            graphs = graphs.to(config['DEVICE'])
            afeats = afeats.to(config['DEVICE'])
            bfeats = bfeats.to(config['DEVICE'])
            labels = labels.to(config['DEVICE'])
            model.zero_grad()
            pred = model(afeats, bfeats, graphs)
            if config['TASK'] == 'regression':
                loss = F.mse_loss(pred, labels)
            elif config['TASK'] == 'classification':
                labels = labels.view(-1)
                pred = pred.view(labels.size(0), -1)
                mask = labels != -1
                labels, pred = labels[mask], pred[mask]
                loss = F.cross_entropy(pred, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
        t1 = time.time()
        print('\t\tEvaluating:')
        if config['TASK'] == 'regression':
            m_eval = evaluate_regression(eval_dtloader, model)
            m_test = evaluate_regression(test_dtloader, model)
        elif config['TASK'] == 'classification':
            m_eval = evaluate_classification(eval_dtloader, model)
            m_test = evaluate_classification(test_dtloader, model)
        if (config['TASK'] == 'regression' and m_eval < best_metrics) or\
            (config['TASK'] == 'classification' and m_eval > best_metrics):
            best_metrics = m_eval
            print(f'\tSaving Model...')
            torch.save(model, "./checkpoints/{}.pt".format(config['DATASET']))
        t2 = time.time()
        print('\tTraining Time: {}'.format(int(t1 - t0)))
        print('\tEvaluating Time: {}'.format(int(t2 - t1)))


if __name__ == "__main__":
    train(config)