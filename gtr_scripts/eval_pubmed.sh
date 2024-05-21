#!/bin/bash

cd ../

# Predefined constants
n_inject_max=400
n_edge_max=25
dataset='pubmed'
embedding='gtr'
eval_embedding='vanilla'  # 'vanilla means directly use data.x, use bow/gtr for transfer'
runs=10
dir='atkg/gtr_norm'
trans='inv'
gpu=1

echo "Running with the following settings:"
echo "Text_type_folder = $save_attack" # {emb}_{norm/vanilla}, if bow, only vanilla
echo "n_inject_max = $n_inject_max"
echo "n_edge_max = $n_edge_max"
echo "dataset = $dataset"
echo "embedding = $embedding"
echo "eval_embedding = $eval_embedding"
echo "dir = $dir"
echo "trans = $trans"
echo "runs = $runs"

# Generate raw texts
python generate_raw_text.py --dir $dir --trans $trans --dataset $dataset

save_attack="$dir\_$trans"


# Eval
echo "BOW!!!"
python gnn_misg.py --dataset $dataset --inductive --eval_robo --model 'gcn' --eval_robo_blk --use_ln 0 --batch_eval --runs $runs --embedding=$embedding \
    --eval_embedding="bow" --save_attack $save_attack --gpu $gpu

echo "BOW!!!"
python gnn_misg.py --dataset $dataset --inductive --eval_robo --model 'gcn' --eval_robo_blk --use_ln 0 --batch_eval --runs $runs --embedding=$embedding \
    --eval_embedding="bow_all" --save_attack $save_attack --gpu $gpu

echo "GTR!!!"
python gnn_misg.py --dataset $dataset --inductive --eval_robo --model 'gcn' --eval_robo_blk --use_ln 0 --batch_eval --runs $runs --embedding=$embedding \
    --eval_embedding="gtr" --save_attack $save_attack --gpu $gpu