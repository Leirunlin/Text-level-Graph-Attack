from collections import Counter
import torch
import torch.nn.functional as F

import attacks.metric as metric
from attacks.seqgia import SEQGIA
from attacks.flipgia import FLIPGIA
from attacks.text_utils import text2emb, raw_data_to_dataset_name


class Attacker(object):
    """
    General Graph Injection Attack Starts from Raw Texts.
    """

    def __init__(self, epsilon, n_epoch, a_epoch, n_inject_max, n_edge_max, feat_lim_min, feat_lim_max, 
                loss=F.nll_loss, eval_metric=metric.eval_acc, device='cpu', early_stop=0, verbose=True, 
                disguise_coe=1.0, sequential_step=0.2, injection="random", branching=False, iter_epoch=2,
                agia_pre=0.5, hinge=False, feat_norm=False, embedding='bow', cooc=False, sp_level=0.2, 
                batch_size=1, feat_upd='flip'):
        self.sequential_step = sequential_step
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
        self.early_stop = early_stop
        self.branching = branching
        self.injection = injection.lower()
        self.iter_epoch = iter_epoch
        self.agia_pre = agia_pre
        self.hinge = hinge
        self.feat_norm = feat_norm
        self.embedding = embedding.lower()
        self.cooc = cooc
        self.sp_level = sp_level
        self.batch_size = batch_size
        self.feat_upd = feat_upd        


    def attack(self, model, adj, features, raw_texts, target_idx, labels=None):
        dataset = raw_data_to_dataset_name(features)
        if self.embedding not in ['gtr']:
            features = text2emb(raw_texts, dataset=dataset, embdding=self.embedding).to(self.device)
        attacker = self._initialize_attacker()
        self.ori_node_num = features.shape[0]
        adj_attack, features_attack = attacker.attack(model, adj, features, target_idx)
        return adj_attack, features_attack


    def _initialize_attacker(self):
        if self.embedding == 'bow':
            return FLIPGIA(
                epsilon=self.epsilon,
                n_epoch=self.n_epoch,
                a_epoch=self.a_epoch,
                n_inject_max=self.n_inject_max,
                n_edge_max=self.n_edge_max,
                feat_lim_min=self.feat_lim_min,
                feat_lim_max=self.feat_lim_max,
                device=self.device,
                early_stop=self.early_stop,
                disguise_coe=self.disguise_coe,
                sequential_step=self.sequential_step,
                injection=self.injection,
                branching=self.branching,
                iter_epoch=self.iter_epoch,
                agia_pre=self.agia_pre,
                verbose=self.verbose,
                cooc=self.cooc,
                sp_level=self.sp_level,
                batch_size=self.batch_size
            )
        elif self.embedding in ['sbert', 'gtr']:
            return SEQGIA(
                epsilon=self.epsilon,
                n_epoch=self.n_epoch,
                a_epoch=self.a_epoch,
                n_inject_max=self.n_inject_max,
                n_edge_max=self.n_edge_max,
                feat_lim_min=self.feat_lim_min,
                feat_lim_max=self.feat_lim_max,
                device=self.device,
                early_stop=self.early_stop,
                disguise_coe=self.disguise_coe,
                sequential_step=self.sequential_step,
                injection=self.injection,
                branching=self.branching,
                iter_epoch=self.iter_epoch,
                agia_pre=self.agia_pre,
                hinge=self.hinge,
                feat_norm=self.feat_norm,
                verbose=self.verbose
            )
    
    