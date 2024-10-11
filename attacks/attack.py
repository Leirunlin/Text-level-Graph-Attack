import random
from attacks import utils
import torch.nn.functional as F
import numpy as np
import torch
import scipy.sparse as sp


def init_feat(num, features, device, style="sample", feat_lim_min=0, feat_lim_max=1):
    if style.lower() == "sample":
        # do random sample from features to init x
        feat_len = features.size(0)
        x = torch.empty((num,features.size(1)),device=features.device)
        sel_idx = torch.randint(0,feat_len,(num,1))
        x = features[sel_idx.view(-1)].clone()
    elif style.lower() == "normal":
        x = torch.randn((num,features.size(1))).to(features.device)
    elif style.lower() == "zeros":
        x = torch.zeros((num,features.size(1))).to(features.device)
    elif style.lower() == "ball":
        directions = torch.randn(num, features.size(1), device=device)
        x = directions / torch.norm(directions, p=2, dim=1, keepdim=True)
    else:
        x = np.random.normal(loc=0, scale=feat_lim_max/10, size=(num, features.size(1)))
        x = utils.feat_preprocess(features=x, device=device)
    return x

# edge-centric cosine similarity analysis
def edge_sim_analysis(edge_index, features):
    sims = []
    for (u,v) in zip(edge_index[0],edge_index[1]):
        sims.append(F.cosine_similarity(features[u].unsqueeze(0),
                                        features[v].unsqueeze(0)).cpu().numpy())
    sims = np.array(sims)
    return sims


def edge_sim_estimate(x, adj, num, style='sample'):
    """
    estimate the mean and variance from the observed data points
    """
    edge_index = adj.coo()[:2]
    sims = edge_sim_analysis(edge_index, x)
    if style.lower() == 'random':
        hs = np.random.choice(sims,size=(num,))
        hs = torch.FloatTensor(hs).to(x.device)
    else:
        mean, var = sims.mean(), sims.var()
        hs = torch.randn((num,)).to(x.device)
        hs = mean + hs*torch.pow(torch.tensor(var),0.5)
    return hs


# node-centric cosine similarity analysis
# analyze 1-hop neighbor cosine similarity
from torch_sparse import SparseTensor, matmul, fill_diag, sum as sparsesum, mul
def gcn_norm(adj_t, order=-0.5, add_self_loops=True):
    if not adj_t.has_value():
        adj_t = adj_t.fill_value(1., dtype=None)
    if add_self_loops:
        adj_t = fill_diag(adj_t, 1.0)
    deg = sparsesum(adj_t, dim=1)
    deg_inv_sqrt = deg.pow_(order)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
    adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
    adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
    return adj_t

def node_sim_analysis(adj, x):
    adj = gcn_norm(adj, add_self_loops=False)
    x_neg = adj @ x
    node_sims = F.cosine_similarity(x_neg, x).cpu().numpy()
    return node_sims


def node_sim_estimate(x, adj, num, style='sample'):
    """
    estimate the mean and variance from the observed data points
    """
    sims = node_sim_analysis(adj, x)
    if style.lower() == 'random':
        hs = np.random.choice(sims, size=(num,))
        hs = torch.FloatTensor(hs).to(x.device)
    else:
        # mean, var = sims.mean(), sims.var()
        # hs = torch.randn((num,)).to(x.device)
        # hs = mean + hs*torch.pow(torch.tensor(var),0.5)
        from scipy.stats import skewnorm
        a, loc, scale = skewnorm.fit(sims)
        hs = skewnorm(a, loc, scale).rvs(num)
        hs = torch.FloatTensor(hs).to(x.device)
    return hs



def compute_homo_loss(data_points, cluster_centers):
    # Computes the mean of the minimum Euclidean distance from each point to the cluster centers
    data_expanded = data_points.unsqueeze(1)  # [n, 1, d]
    centers_expanded = cluster_centers.unsqueeze(0)  # [1, k, d]
    distances = (data_expanded - centers_expanded).pow(2).sum(-1)
    min_distances = distances.min(1)[0]
    return min_distances.mean()


def compute_similarity_loss(features_attack, features, features_propagate=None, adj_attack=None,
                            features_concat=None, n_total=None, homophily=None, hinge=False, disguise_coe=1.0, sim='dis'):
    # Differentiation by maximizing differences (HAO)
    if features_propagate is None:
        with torch.no_grad():
            features_propagate = gcn_norm(adj_attack, add_self_loops=False) @ features_concat
            features_propagate = features_propagate[n_total:]
    sims = F.cosine_similarity(features_attack, features_propagate)
    if homophily is not None:
        # Minimize the distance to sampled homophily
        if hinge:
            # Hinge loss
            mask = sims < homophily
            new_disguise_coe = torch.ones_like(sims)
            new_disguise_coe[mask] = disguise_coe
            new_disguise_coe[~mask] = disguise_coe * 0.5
            homo_loss = (new_disguise_coe * (sims - homophily)).mean()
        else:
            homo_loss = disguise_coe * ((sims - homophily).mean())
    else:
        # Maximize similarity
        homo_loss = disguise_coe * sims.mean()

    return homo_loss


def gia_update_features(attacker, model, adj_attack, features, features_attack, origin_labels, target_idx, sim='dis', homophily=None, hinge=False, feat_norm=False):
    attacker.early_stop.reset()
    if hasattr(attacker, 'disguise_coe'):
        disguise_coe = attacker.disguise_coe
    else:
        disguise_coe = 0
    epsilon = attacker.epsilon
    n_epoch = attacker.n_epoch
    feat_lim_min, feat_lim_max = attacker.feat_lim_min, attacker.feat_lim_max
    n_total = features.shape[0]
    model.eval()

    features_propagate = None
    for i in range(n_epoch):
        features_attack.requires_grad_(True)
        features_concat = torch.cat((features, features_attack), dim=0)
        pred = model(features_concat, adj_attack)
        pred_loss = attacker.loss(pred[:n_total][target_idx], origin_labels[target_idx])
        homo_loss = compute_similarity_loss(features_attack, features, features_propagate, adj_attack, features_concat,
                                            n_total, homophily, hinge, disguise_coe, sim)
        pred_loss += homo_loss * 10 # *1, *3, *5, *7 Weight here
        model.zero_grad()
        pred_loss.backward()
        
        if feat_norm:
            grad = features_attack.grad.data
            features_attack = features_attack.detach() + epsilon * grad.sign()
            features_attack = features_attack / features_attack.norm(dim=1, keepdim=True)
        else:
            grad = features_attack.grad.data
            features_attack = features_attack.detach() + epsilon * grad.sign()
            features_attack = torch.clamp(features_attack, feat_lim_min, feat_lim_max)
       
        """
        # Riemannian gradient descent
        # Cool but not helpful
        if feat_norm:
            with torch.no_grad():
                grad = features_attack.grad
                grad_projected = grad - (grad * features_attack).sum(dim=1, keepdim=True) * features_attack
                features_attack = features_attack + epsilon * grad_projected
                features_attack /= features_attack.norm(dim=1, keepdim=True)
        else:
            grad = features_attack.grad.data
            features_attack = features_attack.detach() + epsilon * grad.sign()
        """

        features_attack = features_attack.detach()
        test_score = attacker.eval_metric(pred[:n_total][target_idx], origin_labels[target_idx])
        if attacker.early_stop:
            attacker.early_stop(test_score)
            if attacker.early_stop.stop:
                print("Attacking: Early stopped.")
                attacker.early_stop.reset()
                return features_attack

        if attacker.verbose:
            print(f"Attacking: Epoch {i}, Loss: {pred_loss.item():.5f}, Surrogate test score: {test_score:.5f}")

    return features_attack


# smooth feature upd from tdgia
def smooth_update_features(attacker, model, adj_attack, features, features_attack, origin_labels, target_idx, homophily=None, n_inject_cur=0, hinge=False, feat_norm=False):
    if hasattr(attacker, 'disguise_coe'):
        disguise_coe = attacker.disguise_coe
    else:
        disguise_coe = 0
    epsilon = attacker.epsilon
    n_epoch = attacker.n_epoch
    #feat_lim_min, feat_lim_max = attacker.feat_lim_min, attacker.feat_lim_max
    # #Bug, if not normalized, arcsin could explode
    feat_lim_max = torch.abs(features_attack).max()
    n_total = features.shape[0]
    model.eval()

    features_attack = features_attack.cpu().data.numpy()
    if feat_lim_max > 1:
        features_attack = features_attack / feat_lim_max
    if features_attack.shape[0] > n_inject_cur:
        features_attack[:-n_inject_cur] = np.arcsin(features_attack[:-n_inject_cur])
    features_attack = utils.feat_preprocess(features=features_attack, device=attacker.device)
    optimizer = torch.optim.Adam([features_attack], lr=epsilon)
    
    features_propagate = None
    for i in range(n_epoch):
        features_attack.requires_grad_(True)
        features_attack_sin = torch.sin(features_attack) * feat_lim_max
        features_concat = torch.cat((features, features_attack_sin), dim=0)

        pred = model(features_concat, adj_attack)
        
        pred_loss = attacker.loss(pred[:n_total][target_idx], origin_labels[target_idx], reduction="none")
        homo_loss = compute_similarity_loss(features_attack, features, features_propagate, adj_attack, features_concat, n_total, homophily, hinge, disguise_coe, sim='hao')
        pred_loss += homo_loss # *1, *3, *5, *7 Weight here
        # Smoothing the loss to prevent gradient explosion
        pred_loss = F.relu(-pred_loss + 5) ** 2
        pred_loss = pred_loss.mean()
        
        optimizer.zero_grad()
        pred_loss.backward(retain_graph=True)
        optimizer.step()

        test_score = attacker.eval_metric(pred[:n_total][target_idx], origin_labels[target_idx])
        if attacker.early_stop:
            attacker.early_stop(test_score)
            if attacker.early_stop.stop:
                print("Attacking: Early stopped.")
                attacker.early_stop.reset()
                return features_attack_sin.detach()

        if attacker.verbose:
            print(f"Attacking: Epoch {i}, Loss: {pred_loss.item():.5f}, Surrogate test score: {test_score:.5f}")

        if torch.isnan(features_attack).any():
            raise ValueError("NaN detected in features_attack after update.")

    if feat_norm:
        features_attack_sin = features_attack_sin / features_attack_sin.norm(dim=1, keepdim=True)
    return features_attack_sin.detach()


def fgsm_update_features(attacker, model, adj_attack, features, features_attack, origin_labels, target_idx, 
                         homophily=None, sparsity_budget=0.2, batch_size=1, hinge=False, cooc=False, 
                         cooc_X=None, verbose=False):
    
    attacker.early_stop.reset()
    model.eval()
    disguise_coe = getattr(attacker, 'disguise_coe', 0)

    features_attack.requires_grad_(True)
    features_attack.retain_grad()
    features_per_row = int(sparsity_budget * features_attack.shape[1])
    n_total = features.shape[0]

    # Initialize the counter for flips per row
    flips_count = (features_attack == 1).sum(dim=1).int()
    his_flips_count = []
    if cooc and batch_size == 1:
        # Cooc lead to instability sometimes
        # Large batch size helps
        batch_size = features_per_row

    while any(flips_count < features_per_row):
        features_concat = torch.cat((features, features_attack), dim=0)
        pred = model(features_concat, adj_attack)
        pred_loss = attacker.loss(pred[:n_total][target_idx],
                                   origin_labels[target_idx], reduction='none')
        pred_loss = pred_loss.mean()
        
        homo_loss = compute_similarity_loss(features_attack, features, None, adj_attack, features_concat,
                                            n_total, homophily, hinge=hinge, disguise_coe=disguise_coe,
                                            sim='hao')
        # Tried homo_loss with 1e-2, 1e-3, 1e-5, not helpful
        # Above 1e-6: only increase homo
        # Below 1e-7: only increase loss
        pred_loss += homo_loss * 1e-6 
        model.zero_grad()
        pred_loss.backward()
        grad = features_attack.grad.data * 1e-5

        with torch.no_grad():
            mask = (flips_count < features_per_row).float().unsqueeze(1).to(grad.device)
            valid_grad = grad * mask
            flip_directions = (grad > 0).float() - (features_attack == 1).float()
            max_loss_grad = valid_grad * flip_directions

            if cooc:
                existing_features = features_attack == 1
                all_zeros_rows = existing_features.sum(dim=1) == 0
                valid_cooc_flips = torch.matmul(existing_features.float(), cooc_X) > 0
                valid_cooc_flips |= ~existing_features  # Allow flips in positions currently at zero
                valid_cooc_flips[all_zeros_rows, :] = True  # Allow all flips in all-zero rows
                max_loss_grad *= valid_cooc_flips

            _, max_indices = torch.topk(max_loss_grad.view(-1), batch_size)
            for idx in max_indices:
                r, c = divmod(idx.item(), features_attack.shape[1])
                if flips_count[r] < features_per_row and (not cooc or cooc_X[r, c]):
                    features_attack[r, c] = 1 - features_attack[r, c]
                    flips_count[r] += 1 if features_attack[r, c] == 1 else -1

            his_flips_count.append(flips_count.sum())
            if len(his_flips_count) > 10 and his_flips_count[-1] <= his_flips_count[-10]:
                # Sometimes entries keeps flipping in cooc
                # We just break
                break

    attacker.early_stop.reset()
    return features_attack




def pgd_update_features(attacker, model, adj_attack, features, features_attack, origin_labels, target_idx, ##
                        homophily=None, sparsity_budget=0.2, cooc=False, batch_size=1, 
                        epochs=200, base_lr=0.1, sample_epochs=20):
    attacker.early_stop.reset()
    model.eval()
    disguise_coe = getattr(attacker, 'disguise_coe', 0)
    features_per_row = int(sparsity_budget * features_attack.shape[1])
    n_total = features.shape[0]

    best_loss = -np.inf
    best_pert = None
    features_attack.requires_grad_(True)
    features_propagate = None

    for t in range(epochs):
        lr = base_lr / np.sqrt(t + 1) * features_per_row
        features_concat = torch.cat((features, features_attack), dim=0)
        pred = model(features_concat, adj_attack)
        pred_loss = attacker.loss(pred[:n_total][target_idx],
                                   origin_labels[target_idx],reduction='none')
        pred_loss = pred_loss.mean()

        homo_loss = compute_similarity_loss(features_attack, features, features_propagate, adj_attack, features_concat,
                                            n_total, homophily, hinge=False, disguise_coe=disguise_coe,
                                            sim='hao')
        pred_loss += homo_loss
        model.zero_grad()
        pred_loss.backward()
        
        # Gradient ascent (maximizing the loss)
        with torch.no_grad():
            grad = features_attack.grad.data
            features_attack.add_(lr * grad)  # Update step
            bisection_projection(features_attack, features_per_row)

    features_attack.detach_()
    for _ in range(sample_epochs):
        sampled = torch.bernoulli(features_attack)
        for i in range(sampled.shape[0]):
            if sampled[i].sum() > features_per_row:
                continue 

        features_concat = torch.cat((features, sampled), dim=0)
        pred = model(features_concat, adj_attack)
        pred_loss = attacker.loss(pred[:n_total][target_idx],
                                   origin_labels[target_idx],reduction='none')
        pred_loss = attacker.loss(pred[:n_total][target_idx],
                                   origin_labels[target_idx],reduction='none')
        pred_loss = pred_loss.mean()

        homo_loss = compute_similarity_loss(features_attack, features, features_propagate, adj_attack, features_concat,
                                            n_total, homophily, hinge=False, disguise_coe=disguise_coe,
                                            sim='hao')
        pred_loss += homo_loss
        
        if pred_loss > best_loss:
            best_loss = pred_loss
            best_pert = sampled.clone()

    if best_pert is not None:
        features_attack[:] = best_pert

    if attacker.verbose:
        print(f"Attacking: Best loss: {best_loss:.5f}")

    return features_attack

def bisection_projection(features, features_per_row):
    with torch.no_grad():
        for i in range(features.shape[0]):
            top = features[i].max().item()
            bot = (features[i].min() - 1).clamp_min(0).item()
            mu = (top + bot) / 2
            while (top - bot) / 2 > 1e-5:
                used_budget = (features[i] - mu).clamp(0, 1).sum()
                if used_budget == features_per_row:
                    break
                elif used_budget > features_per_row:
                    bot = mu
                else:
                    top = mu
                mu = (top + bot) / 2
            features[i].sub_(mu).clamp_(0, 1)