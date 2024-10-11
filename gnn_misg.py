from attacks.attacker import Attacker
from attacks.rnd import RND
from attacks.tga import TGA


from copy import deepcopy
import argparse
import os
import platform

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from torch_geometric.data import Data
from torch_geometric.utils import index_to_mask

from load_graph import generate_grb_split
from models.model_pyg import *
from utils import set_rand_seed, inductive_split, get_index_induc, target_select
from data_preprocess import text2emb
import timeit



def train(model, x, adj_t, y, train_idx, optimizer):
    model.train()

    optimizer.zero_grad()
    out = model(x, adj_t)
    # transductive setting
    if train_idx.size(0) < y.size(0):
        out = out[train_idx]
        y = y[train_idx]
    loss = F.nll_loss(out, y.view(-1))
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test(model, x, adj_t, y, split_idx, evaluator):
    model.eval()

    out = model(x, adj_t)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc

@torch.no_grad()
def sep_test(model, x, adj_t, y, target_idx, evaluator):
    model.eval()
    out = model(x, adj_t)
    out = out[target_idx] if target_idx.size(0) < out.size(0) else out
    y = y[target_idx] if out.size(0) < y.size(0) else y

    y_pred = out.argmax(dim=-1, keepdim=True)
    acc = evaluator.eval({
        'y_true': y,
        'y_pred': y_pred,
    })['acc']
    return acc


def eval_robustness(model, features, adj, target_idx, raw_data, device, args, run):
    # when evaluating robustness in blackbox setting 
    # the attacked graph&data will be loaded from pre-defined path
    raw_texts = raw_data.raw_texts
    labels = raw_data.y
    if args.eval_robo_blk:
        if args.eval_attack.lower()[:3] == 'tga':
            injection = args.eval_attack[4:]
            graph_path = f"{args.save_attack}/tga_{args.prompt}/{args.dataset}_{injection}"
        else:
            graph_path = os.path.join(args.save_attack,args.dataset)+f"_{args.eval_attack}"
        if args.eval_target:
            graph_path += "_target"
        graph_path += f"_0.pt"
        new_data = torch.load(graph_path)
        raw_texts = new_data.raw_texts
        new_data = T.ToSparseTensor()(new_data)
        if args.eval_embedding in ['sbert', 'gtr']:
            feat_attack = text2emb(raw_texts[new_data.y.size(0):], dataset=args.dataset, 
                                   embdding=args.eval_embedding).to(device)
        elif args.eval_embedding in ['bow']:
            # Using Part Bow vocabulary
            new_data.x = text2emb(raw_texts, dataset=args.dataset, embdding=args.eval_embedding, all=False)
            feat_attack = new_data.x[new_data.y.size(0):].to(device)
        elif args.eval_embedding in ['bow_all']:
            # Using All Bow vocabulary
            new_data.x = text2emb(raw_texts, dataset=args.dataset, embdding=args.eval_embedding, all=True)
            feat_attack = new_data.x[new_data.y.size(0):].to(device)
        else:
            # Origin by default
            new_data.x = new_data.x.to_dense()
            feat_attack = new_data.x[new_data.y.size(0):].to(device)

        adj_attack = new_data.adj_t.to(device)
        if args.eval_target:
            target_idx = new_data.target_idx
        return feat_attack, adj_attack, target_idx, raw_texts


    # initialize the corresponding adversary
    if args.eval_attack.lower() == "rnd":
        attacker = RND(epsilon=args.attack_lr,
                 n_epoch=args.attack_epoch,
                 n_inject_max= args.n_inject_max,
                 n_edge_max= args.n_edge_max,
                 feat_lim_min=args.feat_lim_min,
                 feat_lim_max=args.feat_lim_max,
                 embedding=args.embedding,
                 device=device,
                 verbose=False,
                 sp_level=args.sp_level)
    elif args.eval_attack.lower() == 'tga':
        attacker = TGA(epsilon=args.attack_lr,
                 n_epoch=args.attack_epoch,
                 a_epoch=args.agia_epoch,
                 n_inject_max=args.n_inject_max,
                 n_edge_max=args.n_edge_max,
                 device=device,
                 early_stop=args.early_stop,
                 sequential_step=args.sequential_step,
                 injection=args.injection,
                 branching=args.branching,
                 iter_epoch=args.iter_epoch,
                 agia_pre=args.agia_pre,
                 verbose=False, 
                 raw_data=raw_data, 
                 embedding=args.embedding,
                 prompt_type=args.prompt)
    else:
        attacker = Attacker(epsilon=args.attack_lr,
                    n_epoch=args.attack_epoch,
                    a_epoch=args.agia_epoch,
                    n_inject_max= args.n_inject_max,
                    n_edge_max= args.n_edge_max,
                    feat_lim_min=args.feat_lim_min,
                    feat_lim_max=args.feat_lim_max,
                    device=device,
                    early_stop=args.early_stop,
                    disguise_coe=args.disguise_coe,
                    sequential_step=args.sequential_step,
                    injection=args.injection,
                    branching=args.branching,
                    iter_epoch=args.iter_epoch,
                    agia_pre=args.agia_pre,
                    hinge=args.hinge,
                    feat_norm=args.feat_norm,
                    sp_level=args.sp_level,
                    batch_size=args.batch_size,
                    cooc=args.cooc,
                    embedding=args.embedding,
                    verbose=False)


    attack_labels = labels if args.attack_label else None
    if args.eval_target:
        target_idx = target_select(model,adj,features,labels,target_idx,args.target_num)
    if args.eval_attack.lower() in ['tga']:
        adj_attack, features_attack, raw_texts = attacker.attack(model=model,
                                                    adj=adj,
                                                    features=features,
                                                    target_idx=target_idx,
                                                    labels=attack_labels)
    else:
        adj_attack, features_attack = attacker.attack(model=model,
                                                    adj=adj,
                                                    features=features,
                                                    raw_texts=raw_texts,
                                                    target_idx=target_idx,
                                                    labels=attack_labels)
    raw_texts = raw_texts
    return features_attack, adj_attack, target_idx, raw_texts

def reproduction_info():
    # save/print system & device information for reproduction assurability
    if "windows" in platform.system().lower():
        os.system("nvidia-smi")
    else:
        os.system("gpustat")
    print(f"cudatoolkit version: {torch.version.cuda}")

def main():
    parser = argparse.ArgumentParser(description='cig-nn')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--model', type=str, default='gcn')
    parser.add_argument('--dataset',type=str,default='cora')
    parser.add_argument('--grb_mode',type=str,default='full')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=128)
    # put a layer norm right after input
    parser.add_argument('--layer_norm_first', action="store_true")
    # put layer norm between layers or not
    parser.add_argument('--use_ln', type=int,default=0)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--l2decay', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)

    #print device information if set true
    parser.add_argument('--reprod', action="store_true")
    parser.add_argument('--inductive', action="store_true")
    

    # train one model and eval with several attacked graphs
    parser.add_argument('--batch_eval', action="store_true")
    parser.add_argument('--batch_attacks', type=list, default=[])

    # save and load best weights for final evaluation
    parser.add_argument('--best_weights', action="store_true")

    ######################## Adv. Training Setting ####################
    parser.add_argument('--step_size', type=float, default=1e-3)
    parser.add_argument('--m', type=int, default=3)
    parser.add_argument('--attack', type=str, default='vanilla')
    parser.add_argument('--pre_epochs', type=int, default=-1)

    ######################## Robustness Eval Setting ####################
    parser.add_argument('--eval_robo', action="store_true")
    # targeted attack else non-targeted
    parser.add_argument('--eval_target', action="store_true")
    # number of targets in each deg category
    parser.add_argument('--target_num', type=int, default=200) 
    
    # if evaluated in blackbox, the attacked graph will be loaded for evaluation
    parser.add_argument('--eval_robo_blk', action="store_true")
    # the attack method used for evaluation
    parser.add_argument('--eval_attack', type=str, default="pgd")
    # maximum number of injected nodes at 'full' data mode
    # if in other data modes, e.g., 'easy', it shall be 1/3 of that in 'full' mode
    parser.add_argument('--n_inject_max', type=int, default=60)
    # maximum number of edges of the injected (per) node 
    parser.add_argument('--n_edge_max', type=int, default=20)
    # attack feat limit, if not spec_feat_lim, auto calculate from data.x
    parser.add_argument('--spec_feat_lim', action="store_true")
    parser.add_argument('--feat_lim_min', type=float, default=-1.0)
    parser.add_argument('--feat_lim_max', type=float, default=1.0)
    # attack feature update epochs
    parser.add_argument('--attack_epoch', type=int, default=500)
    # attack A_atk update epochs
    parser.add_argument('--agia_epoch', type=int, default=300)
    # how much vicious nodes being injected randomly before agia is applied
    parser.add_argument('--agia_pre', type=float, default=0.5)
    # number of iterative epochs for agia
    parser.add_argument('--iter_epoch', type=int, default=2)
    # attack step size
    parser.add_argument('--attack_lr', type=float, default=0.01)
    # early stopping feat upd for attack
    parser.add_argument('--early_stop', type=int, default=200)
    # weight of the disguised regularization term
    parser.add_argument('--disguise_coe', type=float, default=0.0)
    parser.add_argument('--hinge', action="store_true")
    # update features with label information if set true
    parser.add_argument('--attack_label', action="store_true")
    # save path of the attacked feature and graph
    parser.add_argument('--save_attack', type=str, default="atkg")
    
    # use corresponding subgraph for attack
    parser.add_argument('--prune_graph', action="store_true")

    # paramters for seqgia
    parser.add_argument('--sequential_step', type=float, default=0.2)
    parser.add_argument('--injection', type=str, default="random")
    parser.add_argument('--branching', action="store_true")

    ######################## Misc Setting ####################
    parser.add_argument('--test_freq', type=int, default=1)
    # threshold for homophily defender
    parser.add_argument('--homo_threshold', type=float, default="0.1")
    # enforce grb split
    parser.add_argument('--grb_split', action="store_true")

    ##################### For Flip Attack ####################
    parser.add_argument('--embedding', type=str, default='tfidf', choices=['tfidf', 'sbert', 'bow', 'gtr'])
    parser.add_argument('--eval_embedding', type=str, default='vanilla', choices=['bow', 'sbert', 'vanilla', 'gtr', 'bow_all'])
    parser.add_argument('--sp_level', type=float, default=0.05)
    parser.add_argument('--cooc', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--feat_upd', type=str, default='flip', choices=['flip', 'pgd'])
    
    ##################### For Continuous Attack ####################
    parser.add_argument('--feat_norm', type=int, default=0)

    ##################### For Text Attack ####################
    parser.add_argument('--prompt', type=str, default='sample')
    parser.add_argument('--trans', type=str, default='vanilla')

    args = parser.parse_args()

    args.best_weights = True
    args.inductive = True
    assert args.inductive, "transductive is not supported"
    report_batch = args.batch_attacks
    if not args.batch_eval:
        args.batch_attacks = []
    else:
        if args.eval_target:
            # targeted attack baselines
            args.batch_attacks = ["vanilla","rnd","gia","seqgia","metagia","tdgia","speitml","atdgia","agia","seqagia"]
            report_batch =       ["vanilla","rnd","gia","seqgia","metagia","tdgia","speitml","atdgia","agia","seqagia"]
        elif args.dataset == 'arxiv':
            args.batch_attacks = ["rnd","seqgia","tdgia","atdgia","seqagia"]
            report_batch =       ["rnd","seqgia","tdgia","atdgia","seqagia"]
        elif args.eval_attack.lower() == 'tga':
            args.batch_attacks = ["tga_random", "tga_tdgia", "tga_atdgia", "tga_agia", "tga_meta"]
            report_batch =       ["tga_random", "tga_tdgia", "tga_atdgia", "tga_agia", "tga_meta"]
        elif args.trans.lower() == 'gen':
            args.batch_attacks = ["rnd","seqgia", "metagia","tdgia","atdgia","seqagia"]
            report_batch =       ["rnd","seqgia", "metagia","tdgia","atdgia","seqagia"]
        else:
            # non-target small graphs
            "TODO: attack"
            args.batch_attacks = ["rnd","seqgia","rseqgia", "metagia","rmetagia","tdgia","rtdgia","atdgia","ratdgia","seqagia","seqragia"]
            report_batch =       ["rnd","seqgia","rseqgia", "metagia","rmetagia","tdgia","rtdgia","atdgia","ratdgia","seqagia","seqragia"]

        assert len(report_batch) <= len(args.batch_attacks)
    if args.reprod:
        reproduction_info()
    print(args)
    
    # set rand seed
    set_rand_seed(args.seed)

    # adjust maximum injected nodes
    if args.grb_mode != 'full':
        args.n_inject_max //= 3

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # Eval embedding is used for evaluation for defender (batch_eval)
    # During attack, the embedding seen by attacker is args.embedding
    # Eval_embedding is vanilla, which is gtr by default
    # Logic implicitly embedded into attacks/attacker.py
    if args.eval_embedding == 'vanilla':
        # gtr by default
        data = torch.load(f"./data/{args.dataset}_fixed_gtr.pt")
        if args.embedding != 'gtr':
            data.x = text2emb(data.raw_texts, dataset=args.dataset, embdding=args.embedding)
            data.x = data.x.float()
    elif args.eval_embedding == 'bow_all':
        # gtr by default
        data = torch.load(f"./data/{args.dataset}_fixed_bow.pt")
    else:
        data = torch.load(f"./data/{args.dataset}_fixed_{args.eval_embedding}.pt")

    data = T.ToSparseTensor()(data)
    if args.dataset != 'arxiv':
        train_mask, val_mask, test_mask = generate_grb_split(data, mode=args.grb_mode)
        split_idx = {'train': torch.nonzero(train_mask, as_tuple=True)[0],
            'valid':torch.nonzero(val_mask, as_tuple=True)[0], 
            'test': torch.nonzero(test_mask, as_tuple=True)[0]}
    else:
        tmp_dataset = PygNodePropPredDataset(name='ogbn-arxiv', transform=T.ToSparseTensor(), root="/data/runlin_lei/data")
        split_idx = tmp_dataset.get_idx_split()
        num_nodes = data.x.shape[0]
        train_mask = index_to_mask(split_idx['train'], size=num_nodes)
        val_mask = index_to_mask(split_idx['valid'], size=num_nodes)
        test_mask = index_to_mask(split_idx['test'], size=num_nodes)
        data.y = data.label
        data.category_names = data.class_name
        data.label_names = data.label_name
        del data.label
        del data.class_name
        del data.label_name

    print(data)
    num_classes = data.y.max().item() + 1
    data.y = data.y.unsqueeze(1)
    data.train_mask, data.val_mask, data.test_mask = train_mask, val_mask, test_mask
    args.feat_lim_min = data.x.min().item()
    args.feat_lim_max = data.x.max().item()

    # initialize GNN models
    if args.model.lower() == "sage":
        model = SAGE(data.num_features, args.hidden_channels,
                     num_classes, args.num_layers,
                     args.dropout, layer_norm_first=args.layer_norm_first,
                     use_ln=args.use_ln)
    elif args.model.lower() == 'mlp':
        model = MLP(data.num_features, args.hidden_channels,
                    num_classes, args.num_layers,
                    args.dropout, layer_norm_first=args.layer_norm_first,
                    use_ln=args.use_ln)
    elif 'egnnguard' in args.model.lower():
        threshold = args.homo_threshold
        model = EGCNGuard(data.num_features, args.hidden_channels,
                    num_classes, args.num_layers,
                    args.dropout, layer_norm_first=args.layer_norm_first,
                    use_ln=args.use_ln, threshold=threshold)
    elif args.model.lower() == 'robustgcn':
        model = RobustGCN(data.num_features, args.hidden_channels,
                     num_classes, args.num_layers,
                     args.dropout)
    elif args.model.lower() == "gat":
        heads = 8
        model = GAT(data.num_features, args.hidden_channels,
                    num_classes, args.num_layers,
                    args.dropout, layer_norm_first=args.layer_norm_first,
                    use_ln=args.use_ln, heads=heads)
    else:
        model = GCN(data.num_features, args.hidden_channels,
                    num_classes, args.num_layers,
                    args.dropout, layer_norm_first=args.layer_norm_first,
                    use_ln=args.use_ln)    

    evaluator = Evaluator(name='ogbn-arxiv')
    model = model.to(device)

    train_idx = split_idx['train'].to(device)
    val_idx = split_idx['valid'].to(device)
    test_idx = split_idx['test'].to(device)
    data = data.to(device)
    raw_texts = data.raw_texts
    category_names = data.category_names
    label_names = data.label_names

    if args.inductive:
        # inductive split will automatically use relative ids for splitted graphs
        adj_train, adj_val, adj_test = inductive_split(data.adj_t, split_idx)
        x_train, y_train = data.x[train_idx], data.y[train_idx]
        train_val_idx, _ = torch.sort(torch.cat([train_idx,val_idx],dim=0))
        x_val, y_val = data.x[train_val_idx], data.y[val_idx]
        x_test, y_test = data.x, data.y[test_idx]
        tval_idx_train, tval_idx_val = get_index_induc(train_idx,val_idx)
        tval_idx_train = torch.LongTensor(tval_idx_train).to(device)
        tval_idx_val = torch.LongTensor(tval_idx_val).to(device)
    else:
        adj_train =  adj_val =  adj_test = data.adj_t
        x_train = x_val = x_test = data.x
        y_train = y_val = y_test = data.y
    trains, vals, tests = [], [], []
    robo_tests = []
    batch_robo_tests = {}

    for run in range(args.runs):
        set_rand_seed(run)  # set up seed for reproducibility 
        final_train_acc, best_val, final_test = 0,0,0
        best_weights = None

        if args.epochs > 0:
            model.reset_parameters()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2decay)
        tot_time = 0
        for epoch in range(1, args.epochs + 1):
            start = timeit.default_timer()
            loss = train(model, x_train, adj_train, y_train, train_idx, optimizer)
            if epoch > args.epochs / 2 and epoch % args.test_freq == 0 or epoch == args.epochs:
                if args.inductive:
                    train_acc = sep_test(model,x_train,adj_train,y_train,train_idx,evaluator)
                    val = sep_test(model,x_val,adj_val,y_val,tval_idx_val,evaluator)
                    tst = sep_test(model,x_test,adj_test,y_test,test_idx,evaluator)
                else:
                    train_acc, val, tst = test(model, x_test, adj_test, y_test, split_idx, evaluator)

                if val > best_val :
                    best_val = val
                    final_test = tst
                    final_train_acc = train_acc
                    if args.best_weights:
                        best_weights = deepcopy(model.state_dict())
            stop = timeit.default_timer()
            tot_time += stop-start
        print(f'Run{run} train: {final_train_acc}, val:{best_val}, test:{final_test}')
        print(f'Avg train time {tot_time/args.epochs}')
        trains.append(final_train_acc)
        vals.append(best_val)
        tests.append(final_test)
    
        if args.eval_robo and not args.batch_eval:
            if args.best_weights and args.epochs>0:
                model.load_state_dict(best_weights)
            test_idx = split_idx["test"].to(device)
            
            target_idx = test_idx
            x_attack, adj_attack, target_idx, raw_texts = eval_robustness(model, x_test, adj_test, target_idx, data, device, args, run)    
            x_new = torch.cat([x_test,x_attack],dim=0) if x_attack != None else x_test
            if len(args.save_attack) > 0 and not args.eval_robo_blk:
                if args.eval_attack.lower() in ['tga']:
                    atkg_path = f"{args.save_attack}/{args.eval_attack.lower()}_{args.prompt}/"
                    os.makedirs(atkg_path, exist_ok=True)
                    atkg_path += f'{args.dataset}_{args.injection}'
                else:
                    atkg_path = os.path.join(args.save_attack, args.dataset)+f"_{args.eval_attack}"
                # targeted attack
                if args.eval_target:
                    atkg_path += "_target"
                atkg_path += f"_{run}.pt"
                print(f"saving the generated atkg to {atkg_path}")
                # saving format of the perturbed graph
                adj_row, adj_col = adj_attack.coo()[:2]
                new_data = Data(edge_index=torch.stack([adj_row,adj_col], dim=0),
                                x=x_new,y=data.y)
                new_data.train_mask = data.train_mask
                new_data.val_mask = data.val_mask
                new_data.test_mask= data.test_mask
                new_data.target_idx= target_idx
                new_data.x = new_data.x.to_sparse()
                new_data.raw_texts = raw_texts
                new_data.category_names = category_names
                new_data.label_names = label_names
                torch.save(new_data.cpu(), atkg_path)

            tst = sep_test(model, x_new, adj_attack, data.y, target_idx, evaluator)
            robo_tests.append(tst)
        elif args.batch_eval:
            if args.best_weights and args.epochs>0:
                model.load_state_dict(best_weights)
            target_idx = test_idx
            for (i, atk) in enumerate(args.batch_attacks):
                args.eval_attack = atk
                x_attack, adj_attack, target_idx, _ = eval_robustness(model, x_test, adj_test, target_idx, data, device, args, run)
                x_new = torch.cat([x_test,x_attack],dim=0) if x_attack != None else x_test
                tst = sep_test(model, x_new, adj_attack, data.y, target_idx, evaluator)
                if run == 0:
                    batch_robo_tests[atk] = [tst]
                else:
                    batch_robo_tests[atk].append(tst)
                print(f"Test robustness accuracy under {atk}: {tst}")
                # save gpu memory
                x_attack.cpu()
                adj_attack.cpu()
                target_idx.cpu()
                torch.cuda.empty_cache()

    print('')
    print(f"Average train accuracy: {np.mean(trains)*100:.3f} ± {np.std(trains)*100:.3f}")
    print(f"Average val accuracy: {np.mean(vals)*100:.3f} ± {np.std(vals)*100:.3f}")
    print(f"Average test accuracy: {np.mean(tests)*100:.3f} ± {np.std(tests)*100:.3f}")
    if args.eval_robo and not args.batch_eval:
        print(f"Average test robustness accuracy: {np.mean(robo_tests)*100:.3f} ± {np.std(robo_tests)*100:.3f}")
    elif args.batch_eval:
        print(f"Model: {args.model}, Use LNi: {args.use_ln}_{args.layer_norm_first}")
        for (i,atk) in enumerate(args.batch_attacks):
            print(f"Average test robustness accuracy under {atk}: {np.mean(batch_robo_tests[atk])*100:.3f} ± {np.std(batch_robo_tests[atk])*100:.3f}")
        if report_batch != None:
            print("name: ")
            for (i,atk) in enumerate(report_batch):
                print("{:.5s},".format(atk),end="")
            print()
            print("mean: ")
            for (i,atk) in enumerate(report_batch):
                print("{:.3f},".format(np.mean(batch_robo_tests[atk])*100),end="")
            print()
            print(" std: ")
            for (i,atk) in enumerate(report_batch):
                print("{:.3f},".format(np.std(batch_robo_tests[atk])*100),end="")
            print()
    

if __name__ == "__main__":
    main()
