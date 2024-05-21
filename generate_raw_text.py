import torch
import torch.nn.functional as F
from collections import Counter
import vec2text
import transformers
from data_preprocess import get_gtr_emb, load_vectorizer
import argparse
import os
import numpy as np
import re



def gen(data, dataset, filename, llm):
    dir_name = filename.split(f'{dataset}')[0]
    file = filename.split(f'{dataset}')[1].split(".pt")[0]
    file = dataset + file

    def save_input(data, dataset):
        ori_node_num = data.y.shape[0]
        features_attack = data.x.to_dense()[ori_node_num:, ]
        vectorizer_path = os.path.join("./bow_cache/", f"{dataset}.pkl")
        vec = load_vectorizer(vectorizer_path)
        words = vec.get_feature_names_out()
        used_words = []
        not_used_words = []
        for doc in features_attack:
            used = [words[i] for i in range(len(words)) if doc[i] == 1]
            not_used = [words[i] for i in range(len(words)) if doc[i] == 0]
            used_words.append(used)
            not_used_words.append(not_used)
        used_words = np.array(used_words, dtype=object)
        not_used_words = np.array(not_used_words, dtype=object)
        if not os.path.exists(f"{dir_name}raw"):
            os.makedirs(f"{dir_name}raw")
        np.save(f"{dir_name}raw/{file}_used.npy", used_words)
        np.save(f"{dir_name}raw/{file}_not_used.npy", not_used_words)
    

    def clear_text(raw_text):
        # Use regular expressions to remove "title" in any casing
        raw_text = re.sub(r"\btitle:\s*", "", raw_text, flags=re.IGNORECASE)
        raw_text = re.sub(r"\btitle\b", "", raw_text, flags=re.IGNORECASE)
        raw_text = re.sub(r"\babstract:\s*", "", raw_text, flags=re.IGNORECASE)
        raw_text = re.sub(r"\babstract\b", "", raw_text, flags=re.IGNORECASE)
        raw_text = raw_text.replace("\n", " ")
        # Remove quotation marks
        raw_text = raw_text.replace("\"", "")
        return raw_text


    def extract_number(filename):
        """Extracts number from a filename like 'result_12.txt'."""
        match = re.search(r'(\d+)', filename)
        return int(match.group(1)) if match else 0


    def load_LLM_output(directory):
        extracted_content = []
        if os.path.exists(directory):
            # Filter only .txt files
            txt_files = [f for f in os.listdir(directory) if f.endswith(".txt")]
            # Sort by extracted numeric part
            txt_files.sort(key=extract_number)

            for i, filename in enumerate(txt_files):
                filepath = os.path.join(directory, filename)
                with open(filepath, 'r') as file:
                    content = file.read()
                    # Find the start and end of the relevant section
                    start_index = content.lower().rfind("title")
                    end_index = content.find("=========")
                    if start_index != -1:
                        # Extract the content and trim any extra whitespace
                        section = content[start_index:end_index].strip()
                        if dataset != 'pubmed':
                            section = clear_text(section)
                        else:
                            section = section.replace("\n", " ")
                        extracted_content.append(section)
                    else:
                        section = content[:end_index].strip()
                        if dataset != 'pubmed':
                            section = clear_text(section)
                        else:
                            section = section.replace("\n", " ")
                        extracted_content.append(section)
                if i < 10: # TODO: debug
                    print(filename, extracted_content[-1])
        return extracted_content

    raw_texts = data.raw_texts
    texts = []

    # 1. Save input to used.npy / not_used.npy
    save_input(data, dataset)
    # Save to dir/raw/xxx_used.npy, dir/raw/xxx_not_used.npy
    
    # 2. Use LLM to generate raw text
    # Suppose it is generated in    dir/gpt/
    # TODO: Merge LLM implementation here

    # 3. Load results
    texts = load_LLM_output(f"{dir_name}{file}")
    if len(texts) > 0:
        if dataset.lower() == 'cora':
            assert len(texts) == 60, "Missing content"
        elif dataset.lower() == 'citeseer':
            assert len(texts) == 90, "Missing content"
        elif dataset.lower() == 'pubmed':
            assert len(texts) == 400, "Missing content"
    raw_texts.extend(texts)
    
    return raw_texts


def text_inversion(data):
    # Only for gtr-t5-base model
    # See https://github.com/jxmorris12/vec2text
    # https://github.com/jxmorris12/vec2text/issues/28

    inversion_model = vec2text.models.InversionModel.from_pretrained(
        "ielabgroup/vec2text_gtr-base-st_inversion"
    )
    model = vec2text.models.CorrectorEncoderModel.from_pretrained(
        "ielabgroup/vec2text_gtr-base-st_corrector"
    )

    inversion_trainer = vec2text.trainers.InversionTrainer(
        model=inversion_model,
        train_dataset=None,
        eval_dataset=None,
        data_collator=transformers.DataCollatorForSeq2Seq(
            inversion_model.tokenizer,
            label_pad_token_id=-100,
        ),
    )

    model.config.dispatch_batches = None

    corrector = vec2text.trainers.Corrector(
    model=model,
    inversion_trainer=inversion_trainer,
    args=None,
    data_collator=vec2text.collator.DataCollatorForCorrection(
        tokenizer=inversion_trainer.model.tokenizer
    ),
)
    raw_texts = data.raw_texts
    ori_node_num = data.y.shape[0]
    features_attack = data.x.to_dense()[ori_node_num:, ]
    res = vec2text.invert_embeddings(
        embeddings=features_attack.cuda(),
        corrector=corrector,
        num_steps=20,
    )
    res_embedding = get_gtr_emb(res).to(features_attack.device)
    cosine_similarities = F.cosine_similarity(features_attack, res_embedding, dim=1)
    print(f"Cos: {cosine_similarities.mean():.3f}")
    for i in res:
        print(i)
    raw_texts.extend(res)
    return raw_texts

def main(dataset, directory, trans_type, llm):
    # Extract the embedding type from the directory name
    embedding = directory.split('/')[-1]
    
    # Create a new directory for the modified files
    if trans_type != 'gen':
        new_directory = f"{directory}_{trans_type}"
        if not os.path.exists(new_directory):
            os.makedirs(new_directory)
    else:
        new_directory = f"{directory}_{trans_type}_{llm}"
        if not os.path.exists(new_directory):
            os.makedirs(new_directory)

    # Process each .pt file
    for filename in os.listdir(directory):
        if filename.endswith('.pt') and dataset in filename:
            file_path = os.path.join(directory, filename)
            new_file_path = os.path.join(new_directory, filename)
            
            # Load the data
            print(f"\nLoading {file_path}")
            data = torch.load(file_path)
            ori_node = data.y.shape[0]
            data.raw_texts = data.raw_texts[:ori_node]  # Remove redundant texts
            
            assert (trans_type == "gen" and 'bow' in embedding) or (trans_type == "inv" and 'gtr' in embedding) \
               , "Text inversion only for gtr, generative methods only for 0-1 embeddings"

            # Apply the appropriate transformation
            if trans_type == 'inv':
                text = text_inversion(data)
            elif trans_type == 'gen':
                text = gen(data, dataset, filename=new_file_path, llm=llm)
            else:
                raise NotImplementedError
            
            if len(text) > ori_node:
                data.raw_texts = text                
                torch.save(data, new_file_path)
                print(f"Saved modified data to {new_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process .pt files in a given directory.")
    parser.add_argument("--dataset", type=str, default='cora')
    parser.add_argument("--dir", type=str, default="", help="Directory containing .pt files to process")
    parser.add_argument("--trans_type", type=str, help="Method of emb to text", default='inv', choices=['inv', 'gen'])
    parser.add_argument("--llm", type=str, default='gpt', choices=['gpt', 'gpt_topic', 'llama', 'llama_topic', 
                                                                   'llama_mask', 'llama_topic_mask'])
    args = parser.parse_args()
    
    main(args.dataset, args.dir, args.trans_type, args.llm)