import torch
from torch.utils.data.dataloader import DataLoader
import csv
import pandas as pd
import numpy as np

from data.utils import load_data, chem_regression_collate_fn, chem_classification_collate_fn
from script_check_answer import check_answer

def inference(dataset, set_name='test'):
    print("\t\tInferencing on {}...".format(dataset))
    if 'sars' in dataset:
        task = 'classification'
        custom_collate_fn = chem_classification_collate_fn
        head = (13, 4)
    else:
        task = 'regression'
        custom_collate_fn = chem_regression_collate_fn
    df = pd.read_csv(f'data/csvs/{dataset}-{set_name}.csv')
    values: np.ndarray = df.values
    list_smiles = values[:, 0].astype(np.str)
    properties = []

    model = torch.load(f"./checkpoints/{dataset}.pt") #
    model.to('cuda:0')
    model.eval()
    dtset = load_data(dataset, set_name=set_name)
    loader = DataLoader(dtset, batch_size=128, shuffle=False, collate_fn=custom_collate_fn)
    for (graphs, afeats, bfeats, labels) in loader:
        graphs = graphs.to('cuda:0')
        afeats = afeats.to('cuda:0')
        bfeats = bfeats.to('cuda:0')
        labels = labels.to('cuda:0')
        with torch.no_grad():
            pred = model(afeats, bfeats, graphs)
            if task == 'classification':
                pred = pred.view(-1, head[1])
                _, pred = torch.max(pred, dim=-1)
                pred = pred.view(-1, head[0])
            pred = pred.cpu().numpy().tolist()
            properties.extend(pred)
    with open(f'answer/{dataset}-pred.csv', 'w+', encoding='utf-8', newline='') as fp:
        writer = csv.writer(fp)
        plen = 1 if task == 'regression' else head[0]
        writer.writerow(['smiles'] + [f'target_{i}' for i in range(plen)])
        for s, p in zip(list_smiles, properties):
            writer.writerow([s] + list(p))

if __name__ == "__main__":
    dataset = 'ESOL'
    inference(dataset, 'test')