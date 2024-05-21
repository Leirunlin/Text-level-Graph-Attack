from attacks.text_utils import text_feature_init, text2emb, raw_data_to_dataset_name
from attacks.injection import agia_injection, meta_injection, random_injection, tdgia_injection, atdgia_injection, atdgia_ranking_select

import numpy as np
import torch
import torch.nn.functional as F

import attacks.metric as metric
from attacks.utils import EarlyStop


class TGA(object):
    r"""
    Text then Graph Attack.
    """

    def __init__(self,
                 epsilon,
                 n_epoch,
                 a_epoch,
                 n_inject_max,
                 n_edge_max,
                 loss=F.nll_loss,
                 eval_metric=metric.eval_acc,
                 device='cpu',
                 early_stop=0,
                 verbose=True,
                 sequential_step=0.2,
                 injection="random",
                 branching=False,
                 iter_epoch=2,
                 agia_pre=0.5, 
                 embedding='bow',
                 raw_data=None, 
                 prompt_type='rnd_class'):
        self.sequential_step = sequential_step
        self.device = device
        self.epsilon = epsilon
        self.n_epoch = n_epoch
        self.a_epoch = a_epoch
        self.n_inject_max = n_inject_max
        self.n_edge_max = n_edge_max
        self.loss = loss
        self.eval_metric = eval_metric
        self.verbose = verbose
        self.embedding= embedding

        if early_stop:
            self.early_stop = EarlyStop(patience=early_stop, epsilon=1e-4)
        else:
            self.early_stop = early_stop
        self.branching = branching
        self.injection = injection.lower()
        self.iter_epoch = iter_epoch
        self.agia_pre = agia_pre
        self.raw_data = raw_data
        self.prompt_type = prompt_type

    def attack(self, model, adj, features, target_idx, labels=None):
        model.to(self.device)
        model.eval()

        if labels == None:
            pred_orig = model(features, adj)
            origin_labels = torch.argmax(pred_orig, dim=1)
        else:
            print("Use original labels!")
            pred_orig = model(features, adj)
            origin_labels = labels.view(-1)
        
        prob_pred_orig = torch.exp(model(features, adj)) # map to probability
        self.adj_degs = torch.zeros((self.n_inject_max,)).long()+self.n_edge_max
        features_h = None # No features h
        raw_texts = None
        n_inject_total = 0
        adj_attack = adj
        n_total, n_feat = features.shape

        dataset = raw_data_to_dataset_name(self.raw_data.x)
        raw_texts = text_feature_init(self.raw_data, prob_pred_orig, self.n_inject_max, self.prompt_type)
        features_all = text2emb(raw_texts, dataset=dataset, embdding=self.embedding)
        features_attack = features_all[n_total:, :].to(self.device)

        tot_target_nodes = len(target_idx)                                                      
        """
        Sequential injection
        """
        while n_inject_total < self.n_inject_max:
            
            if n_inject_total > 0:
                with torch.no_grad():
                    current_pred = F.softmax(model(features[:n_total + n_inject_total, :], adj_attack), dim=1)
            else:
                current_pred = pred_orig
            n_inject_cur = min(self.n_inject_max - n_inject_total, max(1, int(self.n_inject_max * self.sequential_step)))
            n_target_cur = min(tot_target_nodes, max(n_inject_cur * (self.n_edge_max + 1), int(tot_target_nodes * self.sequential_step)))
            if self.branching:
                cur_target_idx = atdgia_ranking_select(adj, n_inject_cur, self.n_edge_max, origin_labels, current_pred, target_idx, ratio=n_target_cur / len(target_idx))
            else:
                cur_target_idx = target_idx
            print("Attacking: Sequential inject {}/{} nodes, target {}/{} nodes".format(n_inject_total + n_inject_cur, self.n_inject_max,len(cur_target_idx), len(target_idx)))
            if self.injection == "tdgia":
                adj_attack = tdgia_injection(adj_attack, n_inject_cur, self.n_edge_max, origin_labels, current_pred, cur_target_idx, self.device)
            elif self.injection == "atdgia":
                adj_attack = atdgia_injection(adj_attack, n_inject_cur, self.n_edge_max, origin_labels, current_pred, cur_target_idx, self.device)
            elif self.injection == "meta":
                self.step_size = self.n_edge_max
                features_tmp = features_all[:n_total + n_inject_total, :]
                features_attack_new = features_all[n_total + n_inject_total : n_total + n_inject_total + n_inject_cur, :]
                adj_attack = random_injection(adj_attack, n_inject_cur, self.n_edge_max, cur_target_idx, self.device)
                meta_epoch = max(1,(n_inject_cur // 6) * 1) if self.n_inject_max <= 600 else (n_inject_cur // 60) * 10
                for i in range(meta_epoch):
                    adj_attack = meta_injection(self, model, adj_attack, n_inject_cur, self.n_edge_max, features_tmp, features_attack_new, cur_target_idx, origin_labels, self.device, real_target_idx=target_idx, homophily=features_h)
            elif self.injection[-4:] == "agia":
                if (n_inject_total + n_inject_cur) < int(self.n_inject_max * self.agia_pre):
                    adj_attack = random_injection(adj_attack, n_inject_cur, self.n_edge_max, cur_target_idx, self.device)
                else:
                    if self.injection[0].lower() == "a":
                        # the default approach
                        self.opt = "adam"
                        self.old_reg = False
                    else:
                        raise Exception("Not implemented")

                    features_tmp = features_all[:n_total + n_inject_total, :]
                    features_attack_new = features_all[n_total + n_inject_total : n_total + n_inject_total + n_inject_cur, :]

                    for epoch in range(self.iter_epoch):
                        if epoch == 0:
                            adj_attack = random_injection(adj_attack, n_inject_cur, self.n_edge_max, cur_target_idx, self.device)
                        else:
                            adj_attack = agia_injection(self, model, adj_attack, n_inject_cur, self.n_edge_max, features_tmp, 
                                            features_attack_new, cur_target_idx, origin_labels, self.device, self.opt, old_reg=False, real_target_idx=target_idx, homophily=features_h)
            else:
                adj_attack = random_injection(adj_attack, n_inject_cur, self.n_edge_max, cur_target_idx, self.device)

            n_inject_total += n_inject_cur

        return adj_attack, features_attack, raw_texts
