# LLMGIA

# Setup
Run the following command to install dependencies.
```
pip install -r requirements.txt
```

Create necessary directories:
```
mkdir atkg/
mkdir openai_io/
mkdir openai_io/db
```

Change the api key in api.py to your own.
```
def efficient_openai_text_api(input_text, filename, savepath, sp, ss, api_key="your-key", rewrite = True)
```

# ITGIA

## Embedding-level Attacks
```
cd ./gtr_scripts
mkdir ../atkg/gtr_norm    # The directory to save GTR-embedding attacked graphs
bash run_cora.sh          # Default dir: atkg/gtr_norm
```

To replicate the results combining with HAO, change the weights in attacks/attack.py, 
And run the Embedding-level Attacks.
For example, to set HAO weight=3, you should set:

```
pred_loss += homo_loss * 10 * 3 # In gia_update_features()
pred_loss += homo_loss * 3 # In smooth_update_features()
cd ./gtr_scripts
mkdir ../atkg/gtr_norm30    # The directory to save GTR-embedding attacked graphs
# modify the default dir in run_cora.sh to atkg/gtr_norm30
bash run_cora.sh          # Current dir: atkg/gtr_norm30
```

## Embedding to Text
```
# ITGIA
bash eval_cora.sh
```

# WTGIA
```
# Embedding-level Attacks

cd ./bow_scripts
mkdir -p ../atkg/bow         # The directory to save BoW-embedding attacked graphs
bash run_cora.sh          # Default dir: atkg/bow

# WTGIA

bash eval_cora.sh llama     # args: gpt/gpt_topic/llama/llama_topic/llama_mask/llama_topic_mask
```

# VTGIA
```
# Embedding-level Attacks
cd ./VTGIA_scripts
bash run_cora_mixing.sh   # VTGIA with mixing prompt
```

# Acknowlegement
The repo is based on [GIA-HAO](https://github.com/LFhase/GIA-HAO) and [Graph-LLM](https://github.com/CurryTang/Graph-LLM).

