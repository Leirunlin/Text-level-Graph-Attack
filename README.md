# LLMGIA

# Setup
Run the following command to install dependencies.
```
pip install requirement.txt
```

Create necessary directories:
```
mkdir atkg/
mkdir openai/
mkdir openai/db
```

Change the api key in api.py to your own.
```
def efficient_openai_text_api(input_text, filename, savepath, sp, ss, api_key="your-key", rewrite = True)
```

# ITGIA
```
# Embedding-level Attacks

cd ./gtr_scripts
mkdir ../atkg/gtr_norm    # The directory to save GTR-embedding attacked graphs
bash run_cora.sh          # Default dir: atkg/gtr_norm

# ITGIA
bash eval_cora.sh
```

# WTGIA
```
# Embedding-level Attacks

cd ./bow_scripts
mkdir ../atkg/bow         # The directory to save BoW-embedding attacked graphs
bash run_cora.sh          # Default dir: atkg/bow

# WTGIA
# Currently not complete, see generate_raw_text.py for details
# The text generation part will be integrated soon

bash eval_cora.sh gpt     # args: gpt/gpt_topic/llama/llama_topic
```

# VTGIA
```
# Embedding-level Attacks
cd ./VTGIA_scripts
bash run_cora_mixing.sh   # VTGIA with mixing prompt
```

# Acknowlegement
The repo is based on [GIA-HAO](https://github.com/LFhase/GIA-HAO) and [Graph-LLM](https://github.com/CurryTang/Graph-LLM).

