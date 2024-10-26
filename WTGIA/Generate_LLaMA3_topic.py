from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList, LogitsProcessor
import torch
import time
from tqdm import tqdm
import os
import numpy as np
import sys

class RestrictProcessor(LogitsProcessor):
    def __init__(self, tokenizer, non_target_tokens):
        self.tokenizer = tokenizer
        self.non_target_tokens = non_target_tokens
        all_specified_and_non_specified = set(non_target_tokens)
        self.stopwords = [i for i in range(tokenizer.vocab_size) if i not in all_specified_and_non_specified]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        scores[:, self.non_target_tokens] = -float('Inf')
        return scores

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

def generate_response(model, tokenizer, logits_processor, messages, max_tokens, terminators):
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


def main(prefixs):
    model_id = "model_path"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id 
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    folder_path = 'bow20'

    file_pairs = {}

    for file in os.listdir(folder_path):
        if file == '.DS_Store':
            continue
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

    for prefix in prefixs:
        files = file_pairs[prefix]
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
                if os.path.exists(f'LLAMA3/topic/{prefix}/result_{id}.txt'):
                    continue
                max_rate = 0
                final_use_rate = 0
                final_not_use_rate = 0
                final_word_count = 0
                Results = ''
                Cap = [word.capitalize() for word in not_used_word]
                not_used_word = np.append(not_used_word, Cap)
                not_used_tokens = [tokenizer.encode(word)[1] for word in not_used_word]
                tokens_with_the = [tokenizer.encode(f"the {word}")[2] for word in not_used_word]
                not_used_tokens.extend(tokens_with_the)
                custom_processor = RestrictProcessor(tokenizer, not_used_tokens)
                logits_processor = LogitsProcessorList([custom_processor])
                messages = [
                    {"role": "system", "content": "A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers."},
                    {"role": "user", "content": "There are "+ f"{num_classes} types of paper, which are " + ", ".join(category_names) + ".\n" + "Generate a title and an abstract for paper belongs to one of the given categories.\nEnsure the generated content explicitly contains the following words: "+ ", ".join(f"'{word}'" for word in used_word) + ".\n" + "These words should appear as specified, without using synonyms, plural forms, or other variants.\n" + f"Length limit: {max_words} words." + "\nOutput the TITLE and ABSTRACT without explanation.\nTITLE:...\nABSTRACT:..."}
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
                    if not os.path.exists(f'LLAMA3/topic/{prefix}'):
                        os.makedirs(f'LLAMA3/topic/{prefix}')
                    with open(f'LLAMA3/topic/{prefix}/result_{id}.txt', 'w') as f:
                        f.write(Results)

                # Round 2-n: User feedback and assistant correction
                for _ in range(3):
                    feedback = f"You forgot to use " + ', '.join(f'\'{word}\'' for word in missing_words) + ".\n" + "Output the corrected TITLE and ABSTRACT without explanation.\nTITLE:...\nABSTRACT:..."
                    messages.append({"role":"user", "content": feedback})
                    response = generate_response(model, tokenizer, logits_processor, messages, max_tokens, terminators)
                    use_rate1, use_rate2, missing_words = calculate_usage_rates(response, used_word, not_used_word)
                    print("Use Rate: {:.2f}%".format(use_rate1))
                    print("Not Use Rate: {:.2f}%".format(use_rate2))
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
                        if not os.path.exists(f'LLAMA3/topic/{prefix}'):
                            os.makedirs(f'LLAMA3/topic/{prefix}')
                        with open(f'LLAMA3/topic/{prefix}/result_{id}.txt', 'w') as f:
                            f.write(Results)
                print(f'Finish id {id}. Use rate is: {max_rate}. Word count is: {final_word_count}.')
                use_rates.append(final_use_rate)
                not_use_rates.append(final_not_use_rate)
                word_counts.append(final_word_count)
            print(f'{prefix} Avg Use Rate: {np.mean(use_rates)}')
            print(f'{prefix} Avg Not Use Rate: {np.mean(not_use_rates)}')
            print(f'{prefix} Avg Word Count: {np.mean(word_counts)}')



if __name__ == "__main__":
    prefix_list = sys.argv[1]
    prefixs = eval(prefix_list)
    main(prefixs)