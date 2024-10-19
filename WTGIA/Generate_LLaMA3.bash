#!/bin/bash
mode = "topic"

prefix_lists=(
"['cora_seqgia_0', 'cora_seqagia_0', 'cora_tdgia_0', 'cora_metagia_0', 'cora_atdgia_0', 'cora_rnd_0', 'pubmed_rnd_0']"
"['citeseer_seqgia_0', 'citeseer_seqagia_0', 'citeseer_tdgia_0', 'citeseer_metagia_0', 'citeseer_atdgia_0', 'citeseer_rnd_0', 'pubmed_tdgia_0']"
"['pubmed_seqgia_0', 'pubmed_seqagia_0']"
"['pubmed_metagia_0', 'pubmed_atdgia_0']"
)

gpu_list=("0" "1" "2" "3") 

for i in "${!prefix_lists[@]}"; do
    prefix_list="${prefix_lists[$i]}"
    gpu_id="${gpu_list[$i]}"
    
    CUDA_VISIBLE_DEVICES="$gpu_id" python Generate_LLaMA3_"$mode".py "$prefix_list" &
done

wait
echo "Data generation complete."
