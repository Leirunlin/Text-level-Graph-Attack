import random

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

import attacks.metric as metric
import attacks.utils as utils
from attacks.utils import EarlyStop


class RND(object):
    """
    random edge connection, 
    random feature generation
    """

    def __init__(self,
                 epsilon,
                 n_epoch,
                 n_inject_max,
                 n_edge_max,
                 feat_lim_min,
                 feat_lim_max,
                 loss=F.nll_loss,
                 eval_metric=metric.eval_acc,
                 embedding='bow',
                 device='cpu',
                 early_stop=False,
                 verbose=True,
                 sp_level=0):
        self.device = device
        self.epsilon = epsilon
        self.n_epoch = n_epoch
        self.n_inject_max = n_inject_max
        self.n_edge_max = n_edge_max
        self.feat_lim_min = feat_lim_min
        self.feat_lim_max = feat_lim_max
        self.loss = loss
        self.eval_metric = eval_metric
        self.verbose = verbose
        self.embedding = embedding
        self.sp_level = sp_level

        # Early stop
        if early_stop:
            self.early_stop = EarlyStop(patience=1000, epsilon=1e-4)
        else:
            self.early_stop = early_stop

    def attack(self, model, adj, features, target_idx, labels=None, raw_texts=None):
        model.to(self.device)
        model.eval()
        n_total, n_feat = features.shape
        
        adj_attack = self.injection(adj=utils.tensor_to_adj(adj),
                                    n_inject=self.n_inject_max,
                                    n_node=n_total,
                                    target_idx=target_idx)
        adj_attack = utils.adj_to_tensor(adj_attack).to(target_idx.device)
        if self.embedding == 'bow':
            if self.sp_level == 0:
                self.sp_level = self.avg_sparsity(features)
            features_attack = torch.bernoulli(torch.full((self.n_inject_max, n_feat), self.sp_level))
            print(self.avg_sparsity(features_attack))
        else:
            features_attack = np.random.uniform(low=0, high=1,
                                    size=(self.n_inject_max, n_feat))
            features_attack = torch.FloatTensor(features_attack)
            features_attack_norm = torch.norm(features_attack, p=2, dim=1, keepdim=True)
            features_attack = features_attack / features_attack_norm
        features_attack = torch.FloatTensor(features_attack)
        return adj_attack, features_attack.to(self.device)

    def injection(self, adj, n_inject, n_node, target_idx):
        r"""

        Description
        -----------
        Randomly inject nodes to target nodes.

        Parameters
        ----------
        adj : scipy.sparse.csr.csr_matrix
            Adjacency matrix in form of ``N * N`` sparse matrix.
        n_inject : int
            Number of injection.
        n_node : int
            Number of all nodes.
        target_idx : torch.Tensor
            Mask of attack target nodes in form of ``N * 1`` torch bool tensor.

        Returns
        -------
        adj_attack : scipy.sparse.csr.csr_matrix
            Adversarial adjacency matrix in form of :math:`(N + N_{inject})\times(N + N_{inject})` sparse matrix.

        """

        # test_index = torch.where(target_idx)[0]
        target_idx = target_idx.cpu()
        n_test = target_idx.shape[0]
        new_edges_x = []
        new_edges_y = []
        new_data = []
        for i in range(n_inject):
            islinked = np.zeros(n_test)
            for j in range(self.n_edge_max):
                x = i + n_node

                yy = random.randint(0, n_test - 1)
                while islinked[yy] > 0:
                    yy = random.randint(0, n_test - 1)
                islinked[yy] = 1
                y = target_idx[yy]
                new_edges_x.extend([x, y])
                new_edges_y.extend([y, x])
                new_data.extend([1, 1])

        add1 = sp.csr_matrix((n_inject, n_node))
        add2 = sp.csr_matrix((n_node + n_inject, n_inject))
        adj_attack = sp.vstack([adj, add1])
        adj_attack = sp.hstack([adj_attack, add2])
        adj_attack.row = np.hstack([adj_attack.row, new_edges_x])
        adj_attack.col = np.hstack([adj_attack.col, new_edges_y])
        adj_attack.data = np.hstack([adj_attack.data, new_data])

        return adj_attack


    def avg_sparsity(self, X):
        """
        Calculate average sparsity
        """
        total_elements = X.numel()
        zero_elements = (X == 0).sum().item()
        sparsity = zero_elements / total_elements
        return 1 - sparsity