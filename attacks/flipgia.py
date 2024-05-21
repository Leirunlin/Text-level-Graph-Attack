from attacks.attack import fgsm_update_features, init_feat, node_sim_estimate, pgd_update_features
from attacks.injection import agia_injection, meta_injection, random_class_injection, random_injection, tdgia_injection, tdgia_ranking_select, atdgia_injection, atdgia_ranking_select

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

import attacks.metric as metric
from attacks.utils import EarlyStop


class FLIPGIA(object):
    r"""

    Graph Injection Attack with flipping features.
    Only applicable to binary features.

    """

    def __init__(self,
                 epsilon,
                 n_epoch,
                 a_epoch,
                 n_inject_max,
                 n_edge_max,
                 feat_lim_min,
                 feat_lim_max,
                 loss=F.nll_loss,
                 eval_metric=metric.eval_acc,
                 device='cpu',
                 early_stop=0,
                 verbose=True,
                 disguise_coe=1.0,
                 sequential_step=0.2,
                 cooc=False,
                 feat_upd='flip',
                 sp_level=0.05,
                 batch_size=1,
                 injection="random",
                 branching=False,
                 iter_epoch=2,
                 agia_pre=0.5):
        self.sequential_step = sequential_step
        self.cooc = cooc
        self.device = device
        self.epsilon = epsilon
        self.n_epoch = n_epoch
        self.a_epoch = a_epoch
        self.n_inject_max = n_inject_max
        self.n_edge_max = n_edge_max
        self.feat_lim_min = feat_lim_min
        self.feat_lim_max = feat_lim_max
        self.loss = loss
        self.eval_metric = eval_metric
        self.verbose = verbose
        self.disguise_coe = disguise_coe
        # Early stop
        if early_stop:
            self.early_stop = EarlyStop(patience=early_stop, epsilon=1e-4)
        else:
            self.early_stop = early_stop
        self.branching = branching
        self.injection = injection.lower()
        self.iter_epoch = iter_epoch
        self.agia_pre = agia_pre
        self.sp_level = sp_level
        self.batch_size = batch_size
        if feat_upd == 'flip':
            self.feat_upd_func = fgsm_update_features
        elif feat_upd == 'pgd':
            self.feat_upd_func = pgd_update_features

    def attack(self, model, adj, features, target_idx, labels=None):
        model.to(self.device)
        model.eval()
        
        if labels == None:
            pred_orig = model(features, adj)
            origin_labels = torch.argmax(pred_orig, dim=1)
        else:
            origin_labels = labels.view(-1)

        if self.sp_level == 0:
            self.sp_level = self.avg_sparsity(features)
        if self.cooc:
            # calculate co-occurance matrix
            self.cooc_x = self.compute_cooccurrence_matrix(features)
        else:
            self.cooc_x = None
        

        self.adj_degs = torch.zeros((self.n_inject_max,)).long()+self.n_edge_max
        n_inject_total = 0
        adj_attack = adj
        features_attack = None
        features_h = node_sim_estimate(features,adj,self.n_inject_max)
        tot_target_nodes = len(target_idx)
        """
        Sequential injection
        """
        while n_inject_total < self.n_inject_max:
            
            if n_inject_total>0:
                with torch.no_grad():
                    current_pred = F.softmax(model(torch.cat((features,features_attack),dim=0), adj_attack), dim=1)
            else:
                current_pred = pred_orig
            n_inject_cur = min(self.n_inject_max-n_inject_total,max(1,int(self.n_inject_max * self.sequential_step)))
            n_target_cur = min(tot_target_nodes,max(n_inject_cur*(self.n_edge_max+1),int(tot_target_nodes * self.sequential_step)))
            if self.branching:
                cur_target_idx = atdgia_ranking_select(adj, n_inject_cur, self.n_edge_max, origin_labels, current_pred, target_idx, ratio=n_target_cur/len(target_idx))
            else:
                cur_target_idx = target_idx
            #print("Attacking: Sequential inject {}/{} nodes, target {}/{} nodes".format(n_inject_total + n_inject_cur, self.n_inject_max,len(cur_target_idx),len(target_idx)))
            if self.injection == "tdgia":
                adj_attack = tdgia_injection(adj_attack, n_inject_cur, self.n_edge_max, origin_labels, current_pred, cur_target_idx, self.device)
            elif self.injection == "atdgia":
                adj_attack = atdgia_injection(adj_attack, n_inject_cur, self.n_edge_max, origin_labels, current_pred, cur_target_idx, self.device)
            elif self.injection == "class":
                adj_attack = random_class_injection(adj_attack, n_inject_cur, self.n_edge_max, origin_labels, cur_target_idx, self.device)
            elif self.injection == "meta":
                self.step_size = self.n_edge_max
                features_tmp = torch.cat((features,features_attack),dim=0) if features_attack!=None else features
                adj_attack = random_injection(adj_attack, n_inject_cur, self.n_edge_max, cur_target_idx, self.device)
                meta_epoch = max(1,(n_inject_cur//6)*1) if self.n_inject_max <=600 else (n_inject_cur//60)*10
                for i in range(meta_epoch):
                    features_attack_new = init_feat(n_inject_cur, features, self.device, style="zeros", 
                                            feat_lim_min=self.feat_lim_min, feat_lim_max=self.feat_lim_max)
                    features_attack_new = self.feat_upd_func(self, model, adj_attack, features_tmp, features_attack_new, origin_labels, 
                                                            target_idx, features_h[n_inject_total:n_inject_total+n_inject_cur], 
                                                            sparsity_budget=self.sp_level, cooc=self.cooc, cooc_X=self.cooc_x,
                                                            batch_size=self.batch_size)
                    adj_attack = meta_injection(self, model, adj_attack, n_inject_cur, self.n_edge_max, features_tmp, 
                                            features_attack_new, cur_target_idx, origin_labels, self.device, real_target_idx=target_idx)
            elif self.injection[-4:] == "agia":
                if (n_inject_total+n_inject_cur) < int(self.n_inject_max*self.agia_pre):
                    adj_attack = random_injection(adj_attack, n_inject_cur, self.n_edge_max, cur_target_idx, self.device)
                else:
                    if self.injection[0].lower() == "a":
                        # the default approach
                        self.opt = "adam"
                        self.old_reg = False
                    else:
                        raise Exception("Not implemented")
                    
                    features_tmp = torch.cat((features,features_attack),dim=0) if features_attack!=None else features
                    
                    for epoch in range(self.iter_epoch):
                        if epoch == 0:
                            adj_attack = random_injection(adj_attack, n_inject_cur, self.n_edge_max, cur_target_idx, self.device)
                        else:
                            adj_attack = agia_injection(self, model, adj_attack, n_inject_cur, self.n_edge_max, features_tmp, 
                                            features_attack_new, cur_target_idx, origin_labels, self.device, self.opt, old_reg=False, real_target_idx=target_idx)
                            if self.old_reg:
                                adj_attack = agia_injection(self, model, adj_attack, n_inject_cur, self.n_edge_max, features_tmp, 
                                            features_attack_new, cur_target_idx, origin_labels, self.device, self.opt, old_reg=True, real_target_idx=target_idx)
                            
                        features_attack_new = init_feat(n_inject_cur, features, self.device, style="zeros", 
                                        feat_lim_min=self.feat_lim_min, feat_lim_max=self.feat_lim_max)
                        features_attack_new = self.feat_upd_func(self, model, adj_attack, features_tmp, features_attack_new, origin_labels, target_idx, 
                                                                homophily=features_h[n_inject_total:n_inject_total+n_inject_cur], 
                                                                sparsity_budget=self.sp_level, cooc=self.cooc, cooc_X=self.cooc_x, batch_size=self.batch_size)
            else:
                adj_attack = random_injection(adj_attack, n_inject_cur, self.n_edge_max, cur_target_idx, self.device)

            features_attack_new = init_feat(n_inject_cur, features, self.device, style="zeros", 
                                        feat_lim_min=self.feat_lim_min, feat_lim_max=self.feat_lim_max)
            features_attack = torch.cat((features_attack,features_attack_new),dim=0) if features_attack!=None else features_attack_new
            
            n_inject_total += n_inject_cur
            features_attack = self.feat_upd_func(self, model, adj_attack, features, features_attack, origin_labels, target_idx, 
                                                features_h[:n_inject_total], sparsity_budget=self.sp_level, cooc=self.cooc, 
                                                cooc_X=self.cooc_x, batch_size=self.batch_size, verbose=True)
        
        print(self.avg_sparsity(features_attack), self.sp_level)
        return adj_attack, features_attack

    def avg_sparsity(self, X):
        """
        Calculate average sparsity
        """
        total_elements = X.numel()
        zero_elements = (X == 0).sum().item()
        sparsity = zero_elements / total_elements
        return 1 - sparsity
    
    def compute_cooccurrence_matrix(self, features):
        """
        Compute a binary co-occurrence matrix from the binary feature tensor X.
        
        Returns:
        - torch.Tensor: A binary tensor where entry (i, j) is 1 if features i and j co-occur
          across any samples and 0 otherwise.
        """
        # Compute the dot product of X transpose and X to find co-occurrence
        cooc_matrix = torch.matmul(features.t().float(), features.float())
        cooc_matrix = (cooc_matrix > 0).float()
        print("cooc", cooc_matrix.sum())
        return cooc_matrix