import argparse
import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from utils import load_data

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--dataset', type=str, default='cora',
                    help='Dataset.')  

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

adj, features, labels, idx_train, idx_val, idx_test = load_data(args.dataset)
model = torch.load("../checkpoints/{}.pt".format(args.dataset))

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

model.eval()
test_labels = labels[idx_test].cpu().numpy()
_, embeds = model(features, adj, output_hidden_states=True)
test_embeds = embeds[idx_test].detach().cpu().numpy()

X_tsne = TSNE(init="pca").fit_transform(test_embeds)

plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=test_labels, alpha=0.5)
plt.xticks([])
plt.yticks([])
plt.title("t-SNE projection on {} test set".format(args.dataset))
plt.savefig("../fig/{}.jpg".format(args.dataset))