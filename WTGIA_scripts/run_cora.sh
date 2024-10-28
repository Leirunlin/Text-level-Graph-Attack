#!/bin/bash

cd ../

# Predefined constants
n_inject_max=60
n_edge_max=20
dataset='cora'
embedding='bow'
eval_embedding='vanilla'
sp_level=0     # Only for tf-idf and bow, 0 for avg sp
feat_norm=0    # Only for tf-idf and sbert
cooc=0         # Only for tf-idf and bow
batch_size=1
feat_upd='flip'
runs=10
gpu=0


echo "Running with the following settings:"
echo "n_inject_max = $n_inject_max"
echo "n_edge_max = $n_edge_max"
echo "dataset = $dataset"
echo "embedding = $embedding"
echo "eval_embedding = $eval_embedding"
echo "sp_level = $sp_level"
echo "feat_norm = $feat_norm"
echo "cooc = $cooc"
echo "batch_size = $batch_size"
echo "feat_upd = $feat_upd"
echo "runs = $runs"


eval=false
save_attack="atkg/bow"
# rnd
python -u gnn_misg.py --dataset $dataset --inductive --eval_robo --eval_attack 'rnd' \
    --n_inject_max $n_inject_max --n_edge_max $n_edge_max --grb_mode 'full' --runs 1 \
    --disguise_coe 0 --use_ln 0 --embedding $embedding --eval_embedding $eval_embedding \
    --feat_norm $feat_norm --cooc $cooc --sp_level $sp_level --batch_size $batch_size \
    --feat_upd $feat_upd --gpu $gpu
mv atkg/${dataset}_rnd_0.pt $save_attack/${dataset}_rnd_0.pt

if [ "$eval" = false ]; then
    # seqgia
    python -u gnn_misg.py --dataset $dataset --inductive --eval_robo --eval_attack 'seqgia' \
        --n_inject_max $n_inject_max --n_edge_max $n_edge_max --grb_mode 'full' --runs 1 \
        --disguise_coe 1 --use_ln 0 --branching --embedding $embedding \
        --eval_embedding $eval_embedding --feat_norm $feat_norm --cooc $cooc --sp_level $sp_level \
        --batch_size $batch_size --feat_upd $feat_upd --gpu $gpu

    mv atkg/${dataset}_seqgia_0.pt $save_attack/${dataset}_rseqgia_0.pt

    # seqagia
    python -u gnn_misg.py --dataset $dataset --inductive --eval_robo --eval_attack 'seqgia' \
        --injection 'agia' --n_inject_max $n_inject_max --n_edge_max $n_edge_max \
        --grb_mode 'full' --runs 1 --disguise_coe 0 --use_ln 0 --agia_pre 0 --branching \
        --embedding $embedding --eval_embedding $eval_embedding --feat_norm $feat_norm \
        --sp_level $sp_level --batch_size $batch_size --feat_upd $feat_upd --gpu $gpu
    mv atkg/${dataset}_seqgia_0.pt $save_attack/${dataset}_seqagia_0.pt
    
    python -u gnn_misg.py --dataset $dataset --inductive --eval_robo --eval_attack 'seqgia' \
        --injection 'agia' --n_inject_max $n_inject_max --n_edge_max $n_edge_max \
        --grb_mode 'full' --runs 1 --disguise_coe 1 --use_ln 0 --agia_pre 0 --branching \
        --embedding $embedding --eval_embedding $eval_embedding --feat_norm $feat_norm \
        --cooc $cooc --sp_level $sp_level --batch_size $batch_size --feat_upd $feat_upd --gpu $gpu
    mv atkg/${dataset}_seqgia_0.pt $save_attack/${dataset}_seqragia_0.pt

    # atdgia
    python -u gnn_misg.py --dataset $dataset --inductive --eval_robo --eval_attack 'seqgia' \
        --n_inject_max $n_inject_max --n_edge_max $n_edge_max --grb_mode 'full' --runs 1 \
        --disguise_coe 0 --use_ln 0 --injection 'atdgia' --embedding $embedding \
        --eval_embedding $eval_embedding --feat_norm $feat_norm --cooc $cooc --sp_level $sp_level \
        --batch_size $batch_size --feat_upd $feat_upd --gpu $gpu
        
    mv atkg/${dataset}_seqgia_0.pt $save_attack/${dataset}_atdgia_0.pt

    python -u gnn_misg.py --dataset $dataset --inductive --eval_robo --eval_attack 'seqgia' \
        --n_inject_max $n_inject_max --n_edge_max $n_edge_max --grb_mode 'full' --runs 1 \
        --disguise_coe 1 --use_ln 0 --injection 'atdgia' --embedding $embedding \
        --eval_embedding $eval_embedding --feat_norm $feat_norm --cooc $cooc --sp_level $sp_level \
        --batch_size $batch_size --feat_upd $feat_upd --gpu $gpu
        
    mv atkg/${dataset}_seqgia_0.pt $save_attack/${dataset}_ratdgia_0.pt

    # tdgia
    python -u gnn_misg.py --dataset $dataset --inductive --eval_robo --eval_attack 'seqgia' \
        --n_inject_max $n_inject_max --n_edge_max $n_edge_max --grb_mode 'full' --runs 1 \
        --disguise_coe 0 --use_ln 0 --injection 'tdgia' --embedding $embedding \
        --eval_embedding $eval_embedding --feat_norm $feat_norm --cooc $cooc --sp_level $sp_level \
        --batch_size $batch_size --feat_upd $feat_upd --gpu $gpu
        
    mv atkg/${dataset}_seqgia_0.pt $save_attack/${dataset}_tdgia_0.pt

    python -u gnn_misg.py --dataset $dataset --inductive --eval_robo --eval_attack 'seqgia' \
        --n_inject_max $n_inject_max --n_edge_max $n_edge_max --grb_mode 'full' --runs 1 \
        --disguise_coe 1 --use_ln 0 --injection 'tdgia' --embedding $embedding \
        --eval_embedding $eval_embedding --feat_norm $feat_norm --cooc $cooc --sp_level $sp_level \
        --batch_size $batch_size --feat_upd $feat_upd --gpu $gpu
        
    mv atkg/${dataset}_seqgia_0.pt $save_attack/${dataset}_rtdgia_0.pt


    # metagia
    python -u gnn_misg.py --dataset $dataset --inductive --eval_robo --eval_attack 'seqgia' \
        --injection 'meta' --n_inject_max $n_inject_max --n_edge_max $n_edge_max \
        --grb_mode 'full' --runs 1 --disguise_coe 1 --use_ln 0 --sequential_step 1.0 \
        --embedding $embedding --eval_embedding $eval_embedding --cooc $cooc --feat_norm $feat_norm \
        --sp_level $sp_level --batch_size $batch_size --feat_upd $feat_upd --gpu $gpu
    mv atkg/${dataset}_seqgia_0.pt $save_attack/${dataset}_rmetagia_0.pt
    
    python -u gnn_misg.py --dataset $dataset --inductive --eval_robo --eval_attack 'seqgia' \
        --injection 'meta' --n_inject_max $n_inject_max --n_edge_max $n_edge_max \
        --grb_mode 'full' --runs 1 --disguise_coe 0 --use_ln 0 --sequential_step 1.0 \
        --embedding $embedding --eval_embedding $eval_embedding --cooc $cooc --feat_norm $feat_norm \
        --sp_level $sp_level --batch_size $batch_size --feat_upd $feat_upd --gpu $gpu
    mv atkg/${dataset}_seqgia_0.pt $save_attack/${dataset}_metagia_0.pt

    python -u gnn_misg.py --dataset $dataset  --inductive --eval_robo --eval_attack 'seqgia' \
        --n_inject_max $n_inject_max --n_edge_max $n_edge_max --grb_mode 'full' --runs 1 \
        --disguise_coe 0 --use_ln 0 --branching --embedding $embedding \
        --eval_embedding $eval_embedding --feat_norm $feat_norm --cooc $cooc --sp_level $sp_level \
        --batch_size $batch_size --feat_upd $feat_upd --gpu $gpu
    mv atkg/${dataset}_seqgia_0.pt $save_attack/${dataset}_seqgia_0.pt

fi

# Eval
python gnn_misg.py --dataset $dataset --inductive --eval_robo --model 'gcn' --eval_robo_blk --use_ln 0 \
                   --batch_eval --runs $runs --embedding=$embedding \
                   --eval_embedding=$eval_embedding --cooc $cooc --sp_level $sp_level --feat_norm $feat_norm \
                   --batch_size $batch_size --feat_upd $feat_upd --save_attack $save_attack --gpu $gpu


python gnn_misg.py --dataset $dataset --inductive --eval_robo --model 'egnnguard' --eval_robo_blk --use_ln 0 \
                   --batch_eval --runs $runs --embedding=$embedding \
                   --eval_embedding=$eval_embedding --cooc $cooc --sp_level $sp_level --feat_norm $feat_norm \
                   --batch_size $batch_size --feat_upd $feat_upd --save_attack $save_attack --gpu $gpu