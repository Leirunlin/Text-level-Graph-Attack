#!/bin/bash

cd ../

n_inject_max=400
n_edge_max=25
runs=1
dataset='pubmed'
attack='tga'
grb_mode='full'
prompt='sample'
embedding='gtr'
gpu=0

# rnd
python -u gnn_misg.py --dataset $dataset  --inductive --eval_robo --eval_attack $attack --n_inject_max $n_inject_max --n_edge_max $n_edge_max --grb_mode $grb_mode --injection random --runs $runs --prompt $prompt --disguise_coe 0 --use_ln 0 --embedding $embedding --gpu $gpu
python -u gnn_misg.py --dataset $dataset  --inductive --eval_robo --eval_attack $attack --n_inject_max $n_inject_max --n_edge_max $n_edge_max --grb_mode $grb_mode --injection tdgia --runs $runs --prompt $prompt --disguise_coe 0 --use_ln 0 --embedding $embedding --gpu $gpu
python -u gnn_misg.py --dataset $dataset  --inductive --eval_robo --eval_attack $attack --n_inject_max $n_inject_max --n_edge_max $n_edge_max --grb_mode $grb_mode --injection atdgia --runs $runs --prompt $prompt --disguise_coe 0 --use_ln 0 --embedding $embedding --gpu $gpu
python -u gnn_misg.py --dataset $dataset  --inductive --eval_robo --eval_attack $attack --n_inject_max $n_inject_max --n_edge_max $n_edge_max --grb_mode $grb_mode --injection agia --runs $runs --prompt $prompt --disguise_coe 0 --use_ln 0 --agia_pre 0 --branching --embedding $embedding --gpu $gpu
python -u gnn_misg.py --dataset $dataset  --inductive --eval_robo --eval_attack $attack --n_inject_max $n_inject_max --n_edge_max $n_edge_max --grb_mode $grb_mode --injection meta --runs $runs --prompt $prompt --disguise_coe 0 --use_ln 0 --sequential_step 1.0 --embedding $embedding --gpu $gpu


runs=10
#### Defense Benchmark ####
# without LNi
echo "BOW!!!"
python gnn_misg.py --dataset $dataset  --inductive --eval_attack $attack --eval_robo   --model 'gcn' --eval_robo_blk --use_ln 0 --batch_eval --runs $runs --prompt $prompt --embedding $embedding --eval_embedding="bow" --gpu $gpu

echo "GTR!!!"
python gnn_misg.py --dataset $dataset  --inductive --eval_attack $attack --eval_robo   --model 'gcn' --eval_robo_blk --use_ln 0 --batch_eval --runs $runs --prompt $prompt --embedding $embedding --eval_embedding="gtr" --gpu $gpu