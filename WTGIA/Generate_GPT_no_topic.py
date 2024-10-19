from openai import OpenAI
import os
import time
import numpy as np
from tqdm import tqdm
import sys

def generate_response(client, prompt, max_tokens):
    response = client.chat.completions.create(
    model="gpt-3.5-turbo-1106",
    messages = prompt,
    max_tokens = max_tokens
    )

    return response.choices[0].message.content

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

def main(client, prefixs):
    system_mes = 'A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.'
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
        if 'not_used' in files and 'used' in files:
            file_t1 = time.time()
            not_used_file = files['not_used']
            used_file = files['used']
            print(f"Processing File Pairs: {not_used_file} and {used_file}")
            used_words = np.load(used_file,  allow_pickle=True)
            not_used_words = np.load(not_used_file,  allow_pickle=True)
            if 'cora' in used_file:
                max_tokens = 512
                max_words = 300
            elif 'citeseer' in used_file:
                max_tokens = 512
                max_words = 300
            elif 'pubmed' in used_file:
                max_tokens = 550
                max_words = 400
            for id, (used_word, not_used_word) in tqdm(enumerate(zip(used_words, not_used_words))):
                if os.path.exists(f'GPT3.5/no_topic/{prefix}/result_{id}.txt'):
                    continue
                max_rate = 0
                message = [{"role": "system", "content": system_mes}]
                # Round 1: Initial request
                content = "Generate a title and an abstract for an academic article.\n" + "Ensure the generated content explicitly contains the following words: " + ", ".join(f"'{word}'" for word in used_word) + ".\nThese words should appear as specified, without using synonyms, plural forms, or other variants.\n" + f"Length limit: {max_words} words." + "\nOutput the TITLE and ABSTRACT without explanation.\nTITLE:...\nABSTRACT:..."
                message.append({"role":"user", "content": content})
                response = generate_response(client, message, max_tokens)
                use_rate1, use_rate2, missing_words = calculate_usage_rates(response, used_word, not_used_word)
                print("Initial Use Rate: {:.2f}%".format(use_rate1))
                print("Initial Not Use Rate: {:.2f}%".format(use_rate2))
                message.append({"role":"assistant", "content": response})
                if use_rate1 >= max_rate:
                    max_rate = use_rate1
                    Results = response
                    Results += '\n\n====================================\n\n'
                    Results += "Should Use Rate: {:.2f}%\n".format(use_rate1)
                    Results += "Should Not Use Rate: {:.2f}%\n".format(use_rate2)
                    Results += f"Word Count: {len(response.split())}"
                    if not os.path.exists(f'GPT3.5/no_topic/{prefix}'):
                        os.makedirs(f'GPT3.5/no_topic/{prefix}')
                    with open(f'GPT3.5/no_topic/{prefix}/result_{id}.txt', 'w') as f:
                        f.write(Results)
            
                # Round 2-n: User feedback and assistant correction
                for _ in range(3):
                    feedback = f"You forgot to use " + ', '.join(f'\'{word}\'' for word in missing_words) + ".\n" + "Output the corrected TITLE and ABSTRACT without explanation.\nTITLE:...\nABSTRACT:..."
                    message.append({"role":"user", "content": feedback})
                    response = generate_response(client, message, max_tokens)
                    use_rate1, use_rate2, missing_words = calculate_usage_rates(response, used_word, not_used_word)
                    print("Use Rate: {:.2f}%".format(use_rate1))
                    print("Not Use Rate: {:.2f}%".format(use_rate2))
                    message.append({"role":"assistant", "content": response})
                    if use_rate1 >= max_rate:
                        max_rate = use_rate1
                        Results = response
                        Results += '\n\n====================================\n\n'
                        Results += "Should Use Rate: {:.2f}%\n".format(use_rate1)
                        Results += "Should Not Use Rate: {:.2f}%\n".format(use_rate2)
                        Results += f"Word Count: {len(response.split())}"
                        if not os.path.exists(f'GPT3.5/no_topic/{prefix}'):
                            os.makedirs(f'GPT3.5/no_topic/{prefix}')
                        with open(f'GPT3.5/no_topic/{prefix}/result_{id}.txt', 'w') as f:
                            f.write(Results)
                print(f'Finish id {id}. Use rate is: {max_rate}')
            file_t2 = time.time()
            print(f'{prefix} Full Time:{file_t2 - file_t1}')


        

if __name__ == "__main__":
    os.environ['OPENAI_API_KEY'] = "your_api_key"
    client = OpenAI()
    prefix_list = sys.argv[1]
    prefixs = eval(prefix_list)
    main(client, prefixs)
