#!/bin/bash

cd ../

# Predefined constants
n_inject_max=500
n_edge_max=30
dataset='reddit'
embedding='bow'
eval_embedding='vanilla'  # 'vanilla means directly use data.x, use bow/gtr for transfer'
runs=10
dir='atkg/bow'
trans='gen'
llm=$1
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
python generate_raw_text.py --dir $dir --trans $trans --dataset $dataset --llm $llm

save_attack="$dir"_"$trans"_"$llm"


# Eval
#echo "SBERT!!!"
#python gnn_misg.py --dataset $dataset --inductive --eval_robo --model 'gcn' --eval_robo_blk --use_ln 0 --batch_eval --runs $runs --embedding=$embedding \
#    --eval_embedding="sbert" --save_attack $save_attack --gpu $gpu --trans $trans

echo "BOW!!!"
python gnn_misg.py --dataset $dataset --inductive --eval_robo --model 'gcn' --eval_robo_blk --use_ln 0 --batch_eval --runs $runs --embedding=$embedding \
    --eval_embedding="bow" --save_attack $save_attack --gpu $gpu --trans $trans

echo "GTR!!!"
python gnn_misg.py --dataset $dataset --inductive --eval_robo --model 'gcn' --eval_robo_blk --use_ln 0 --batch_eval --runs $runs --embedding=$embedding \
    --eval_embedding="gtr" --save_attack $save_attack --gpu $gpu --trans $trans



