import torch
import torch.nn.functional as F
from collections import Counter
import vec2text
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList, LogitsProcessor
from tqdm import tqdm
from data_preprocess import get_gtr_emb, load_vectorizer
import argparse
import os
import numpy as np
import re

class RestrictProcessor(LogitsProcessor):
    def __init__(self, tokenizer, non_target_tokens):
        self.tokenizer = tokenizer
        self.non_target_tokens = non_target_tokens
        all_specified_and_non_specified = set(non_target_tokens)
        self.stopwords = [i for i in range(tokenizer.vocab_size) if i not in all_specified_and_non_specified]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        scores[:, self.non_target_tokens] = -float('Inf')
        return scores

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
                with open(filepath, 'r', encoding = 'utf-8') as file:
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
    
    def calculate_usage_rates(text, should_use_words, should_not_use_words):
        text = text.lower().split()
        text_words = [subpart for part in text for subpart in part.split('-')]
        non_use = []
        
        should_use_count = 0
        should_not_use_count = 0
        
        for word in should_use_words:
            if word in text_words:
                should_use_count += 1
            else:
                non_use.append(word)
        for word in should_not_use_words:
            if word in text_words:
                should_not_use_count += 1

        should_use_rate = (should_use_count / len(should_use_words)) * 100 if len(should_use_words) > 0 else 0
        should_not_use_rate = (should_not_use_count / len(should_not_use_words)) * 100 if len(should_not_use_words) > 0 else 0
        
        return should_use_rate, should_not_use_rate, non_use

    def generate_response_gpt(client, prompt, max_tokens):
        response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106", # model_name
        messages = prompt,
        max_tokens = max_tokens
        )

        return response.choices[0].message.content 

    def generate_response_llama(model, tokenizer, logits_processor, messages, max_tokens, terminators):
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)

        outputs = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            eos_token_id=terminators,
            pad_token_id=128001,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            logits_processor=logits_processor
        )
        response = outputs[0][input_ids.shape[-1]:]
        text = tokenizer.decode(response, skip_special_tokens=True)

        return text
    
    def generate(dir_name, llm):
        if 'gpt' in llm:
            generate_response = generate_response_gpt
        else:
            generate_response = generate_response_llama
        model_path = args.model_path
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            # attn_implementation="flash_attention_2" 
        )
        tokenizer.pad_token_id = tokenizer.eos_token_id  
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        folder_path = f'{dir_name}raw'
        file_pairs = {}

        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path):
                if file.endswith('not_used.npy'):
                    suffix = 'not_used'
                    prefix = file[:-len('not_used.npy') - 1] 
                elif file.endswith('used.npy'):
                    suffix = 'used'
                    prefix = file[:-len('used.npy') - 1] 

                if prefix not in file_pairs:
                    file_pairs[prefix] = {}
                file_pairs[prefix][suffix] = file_path

        for prefix, files in file_pairs.items():
            if not os.path.exists(f'{dir_name}{llm}/{prefix}'):
                os.makedirs(f'{dir_name}{llm}/{prefix}')
            use_rates = []
            not_use_rates = []
            word_counts = []
            if 'not_used' in files and 'used' in files:
                not_used_file = files['not_used']
                used_file = files['used']
                print(f"Processing File Pairs: {not_used_file} and {used_file}")
                used_words = np.load(used_file,  allow_pickle=True)
                not_used_words = np.load(not_used_file,  allow_pickle=True)
                if 'cora' in used_file:
                    max_tokens = 512
                    max_words = 300
                    num_classes = 7
                    category_names = ["Rule Learning", "Neural Networks", "Case Based", "Genetic Algorithms", "Theory", "Reinforcement Learning", "Probabilistic Methods"]
                elif 'citeseer' in used_file:
                    max_tokens = 512
                    max_words = 300
                    num_classes = 6
                    category_names = ["Agents", "Machine Learning", "Information Retrieval", "Database", "Human Computer Interaction", "Artificial Intelligence"]
                elif 'pubmed' in used_file:
                    max_tokens = 550
                    max_words = 400
                    num_classes = 3
                    category_names = ['Diabetes Mellitus Experimental', 'Diabetes Mellitus Type 1', 'Diabetes Mellitus Type 2']
                for id, (used_word, not_used_word) in tqdm(enumerate(zip(used_words, not_used_words))):
                    if os.path.exists(f'{dir_name}{llm}/{prefix}/result_{id}.txt'):
                        continue
                    max_rate = 0
                    final_use_rate = 0
                    final_not_use_rate = 0
                    final_word_count = 0
                    Results = ''
                    Cap = [word.capitalize() for word in not_used_word]
                    not_used_word = np.append(not_used_word, Cap)
                    not_used_tokens = [tokenizer.encode(word)[1] for word in not_used_word]
                    the_not_used_tokens = [tokenizer.encode(f"the {word}")[2] for word in not_used_word]
                    not_used_tokens.extend(the_not_used_tokens)
                    if 'mask' in llm: 
                        custom_processor = RestrictProcessor(tokenizer, [])
                    else: # no restrict
                        custom_processor = RestrictProcessor(tokenizer, not_used_tokens)
                    logits_processor = LogitsProcessorList([custom_processor])
                    if 'topic' in llm: 
                        messages = [
                            {"role": "system", "content": "A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers."},
                            {"role": "user", "content": "There are "+ f"{num_classes} types of paper, which are " + ", ".join(category_names) + ".\n" + "Generate a title and an abstract for paper belongs to one of the given categories.\nEnsure the generated content explicitly contains the following words: "+ ", ".join(f"'{word}'" for word in used_word) + ".\n" + "These words should appear as specified, without using synonyms, plural forms, or other variants.\n" + f"Length limit: {max_words} words." + "\nOutput the TITLE and ABSTRACT without explanation.\nTITLE:...\nABSTRACT:..."}
                        ]
                    else:
                        messages = [
                            {"role": "system", "content": "A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers."},
                            {"role": "user", "content": "Generate a title and an abstract for an academic article.\n" + "Ensure the generated content explicitly contains the following words: "+ ", ".join(f"'{word}'" for word in used_word) + ".\n" + "These words should appear as specified, without using synonyms, plural forms, or other variants.\n" + f"Length limit: {max_words} words." + "\nOutput the TITLE and ABSTRACT without explanation.\nTITLE:...\nABSTRACT:..."}
                        ]
                    # Round 1: Initial request
                    response = generate_response(model, tokenizer, logits_processor, messages, max_tokens, terminators)
                    use_rate1, use_rate2, missing_words = calculate_usage_rates(response, used_word, not_used_word)
                    print("Initial Use Rate: {:.2f}%".format(use_rate1))
                    print("Initial Not Use Rate: {:.2f}%".format(use_rate2))
                    messages.append({"role":"assistant", "content": response})
                    if use_rate1 >= max_rate:
                        max_rate = use_rate1
                        final_use_rate = use_rate1
                        final_not_use_rate = use_rate2
                        Results = response
                        Results += '\n\n====================================\n\n'
                        Results += "Should Use Rate: {:.2f}%\n".format(use_rate1)
                        Results += "Should Not Use Rate: {:.2f}%\n".format(use_rate2)
                        Results += f"Word Count: {len(response.split())}"
                        final_word_count = len(response.split())
                        with open(f'{dir_name}{llm}/{prefix}/result_{id}.txt', 'w') as f:
                            f.write(Results)

                    # Round 2-n: User feedback and assistant correction
                    for _ in range(3):
                        feedback = f"You forgot to use " + ', '.join(f'\'{word}\'' for word in missing_words) + ".\n" + "Output the corrected TITLE and ABSTRACT without explanation.\nTITLE:...\nABSTRACT:..."
                        messages.append({"role":"user", "content": feedback})
                        response = generate_response(model, tokenizer, logits_processor, messages, max_tokens, terminators)
                        use_rate1, use_rate2, missing_words = calculate_usage_rates(response, used_word, not_used_word)
                        messages.append({"role":"assistant", "content": response})
                        if use_rate1 >= max_rate:
                            max_rate = use_rate1
                            final_use_rate = use_rate1
                            final_not_use_rate = use_rate2
                            Results = response
                            Results += '\n\n====================================\n\n'
                            Results += "Should Use Rate: {:.2f}%\n".format(use_rate1)
                            Results += "Should Not Use Rate: {:.2f}%\n".format(use_rate2)
                            Results += f"Word Count: {len(response.split())}"
                            final_word_count = len(response.split())
                            with open(f'{dir_name}{llm}/{prefix}/result_{id}.txt', 'w') as f:
                                f.write(Results)
                    print(f'Finish id {id}. Use rate is: {max_rate}. Word count is: {final_word_count}.')
                    use_rates.append(final_use_rate)
                    not_use_rates.append(final_not_use_rate)
                    word_counts.append(final_word_count)
                print(f'{prefix} Avg Use Rate: {np.mean(use_rates)}')
                print(f'{prefix} Avg Not Use Rate: {np.mean(not_use_rates)}')
                print(f'{prefix} Avg Word Count: {np.mean(word_counts)}')

    raw_texts = data.raw_texts
    texts = []

    # 1. Save input to used.npy / not_used.npy
    save_input(data, dataset)
    # Save to dir_name/raw/xxx_used.npy, dir_name/raw/xxx_not_used.npy
    
    # 2. Use LLM to generate raw text
    generate(dir_name, llm)   
    # Save to dir_name/llm/file

    # 3. Load results
    texts = load_LLM_output(f"{dir_name}{llm}/{file}")
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
    parser.add_argument("--dir", type=str, default="atkg/bow", help="Directory containing .pt files to process")
    parser.add_argument("--trans_type", type=str, help="Method of emb to text", default='gen', choices=['inv', 'gen'])
    parser.add_argument("--llm", type=str, default='llama', choices=['gpt', 'gpt_topic', 'llama', 'llama_topic', 
                                                                   'llama_mask', 'llama_topic_mask'])
    parser.add_argument("--model_path", type=str, default='meta-llama/Meta-Llama-3-8B')
    args = parser.parse_args()
    
    main(args.dataset, args.dir, args.trans_type, args.llm)