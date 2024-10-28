"""
A general graph dataset loading component
"""
import torch
import os
import numpy as np
import torch_geometric.transforms as T
import platform
import scipy.sparse as sp
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops
from torch_geometric.datasets import Planetoid, CitationFull
import torch_geometric.transforms as T


if "windows" in platform.system().lower():
    base_dir = "E:/.datasets"
else:
    base_dir = "../.datasets"


class Mask(object):
    def __init__(self, train_mask, val_mask, test_mask):
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask

def generate_percent_split(dataset, seed=0, train_percent=10, val_percent=10):
    data = dataset[0]
    num_classes = dataset.num_classes
    train_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
    val_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
    test_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
    for c in range(num_classes):
        all_c_idx = torch.nonzero(data.y == c,as_tuple=True)[0].flatten()
        num_c = all_c_idx.size(0)
        train_num_per_c = num_c * train_percent // 100
        val_num_per_c = num_c * val_percent // 100
        perm = torch.randperm(all_c_idx.size(0))
        c_train_idx = all_c_idx[perm[:train_num_per_c]]
        train_mask[c_train_idx] = True
        test_mask[c_train_idx] = True
        c_val_idx = all_c_idx[perm[train_num_per_c : train_num_per_c + val_num_per_c]]
        val_mask[c_val_idx] = True
        test_mask[c_val_idx] = True
    test_mask = ~test_mask
    return train_mask, val_mask, test_mask

def generate_split(dataset, seed=0, train_num_per_c=20, val_num_per_c=30):
    data = dataset[0]
    num_classes = dataset.num_classes
    train_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
    val_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
    test_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
    for c in range(num_classes):
        all_c_idx = (data.y == c).nonzero()
        if all_c_idx.size(0) <= train_num_per_c + val_num_per_c:
            test_mask[all_c_idx] = True
            continue
        perm = torch.randperm(all_c_idx.size(0))
        c_train_idx = all_c_idx[perm[:train_num_per_c]]
        train_mask[c_train_idx] = True
        test_mask[c_train_idx] = True
        c_val_idx = all_c_idx[perm[train_num_per_c : train_num_per_c + val_num_per_c]]
        val_mask[c_val_idx] = True
        test_mask[c_val_idx] = True
    test_mask = ~test_mask
    return train_mask, val_mask, test_mask

def generate_grb_split(data,mode='full'):
    # data = dataset[0]
    # num_classes = dataset.num_classes
    train_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
    val_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
    test_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
    adj = data.adj_t
    degs = adj.sum(-1)
    _, idx = degs.sort()
    n_total = len(idx)
    n_out = int(n_total*0.05)
    n_cag = int((n_total-n_out)*0.3)
    easy_idx = idx[n_out:n_out+n_cag]
    med_idx = idx[n_out+n_cag:n_out+n_cag+n_cag]
    hard_idx = idx[n_out+n_cag+n_cag:n_out+n_cag+n_cag+n_cag]
    esel_idx = torch.randperm(n_cag)
    msel_idx = torch.randperm(n_cag)
    hsel_idx = torch.randperm(n_cag)
    n_test = int(n_total*0.1)

    if mode.lower() == "full":
        test_mask[easy_idx[esel_idx[:n_test]]] = 1
        test_mask[med_idx[msel_idx[:n_test]]] = 1
        test_mask[hard_idx[hsel_idx[:n_test]]] = 1
    elif mode.lower() == "easy":
        test_mask[easy_idx[esel_idx[:n_test]]] = 1
    elif mode.lower() == "medium":
        test_mask[med_idx[msel_idx[:n_test]]] = 1
    elif mode.lower() == "hard":
        test_mask[hard_idx[hsel_idx[:n_test]]] = 1
    else:
        raise Exception("no such mode")
    left_idx = torch.nonzero(torch.logical_not(torch.logical_or(test_mask,train_mask)),as_tuple=True)[0]
    random_idx = torch.randperm(len(left_idx))
    n_train = int(len(idx)*0.6)
    train_mask[left_idx[random_idx[:n_train]]] = 1
    val_mask[left_idx[random_idx[n_train:]]] = 1
    print(f"split datasets into train {train_mask.sum()}/{n_total}, deg {degs[train_mask].mean()}")
    print(f"                      val {val_mask.sum()}/{n_total}, deg {degs[val_mask].mean()}")
    print(f"                     test {test_mask.sum()}/{n_total}, deg {degs[test_mask].mean()}")
    return train_mask, val_mask, test_mask

def load_split(path):
    mask = torch.load(path)
    return mask.train_mask, mask.val_mask, mask.test_mask
