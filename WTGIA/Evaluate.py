import os
import numpy as np
from tqdm import tqdm
import re

def main():
    prefixs = ['cora_seqgia_0', 'cora_seqagia_0', 'cora_tdgia_0', 'cora_metagia_0', 'cora_atdgia_0', 'cora_rnd_0',
               'citeseer_seqgia_0', 'citeseer_seqagia_0', 'citeseer_tdgia_0', 'citeseer_metagia_0', 'citeseer_atdgia_0', 'citeseer_rnd_0',
               'pubmed_seqgia_0', 'pubmed_seqagia_0', 'pubmed_tdgia_0', 'pubmed_metagia_0', 'pubmed_atdgia_0', 'pubmed_rnd_0']
    all_results = ''
    folder_path = 'bow20'
    mode = 'no_topic'

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
            for id,_ in tqdm(enumerate(zip(used_words, not_used_words))):
                if os.path.exists(f'LLAMA3/{mode}/{prefix}/result_{id}.txt'):
                    with open(f'LLAMA3/{mode}/{prefix}/result_{id}.txt') as f:
                        output = f.read()
                        
                        pattern_explicit = r'Should Use Rate: (\d+\.\d+)%|Should Not Use Rate: (\d+\.\d+)%|Word Count: (\d+)'
                        
                        matches_explicit = re.findall(pattern_explicit, output)

                        use_rate = next((match[0] for match in matches_explicit if match[0]), None)
                        not_use_rate = next((match[1] for match in matches_explicit if match[1]), None)
                        count = next((match[2] for match in matches_explicit if match[2]), None)

                        use_rates.append(float(use_rate))
                        not_use_rates.append(float(not_use_rate))
                        word_counts.append(float(count))
            all_results += f'{prefix} {mode}:' + '\n'
            all_results += 'Avg Use Rate: {:.2f}%\n'.format(np.mean(use_rates))
            all_results += 'Avg Not Use Rate: {:.2f}%\n'.format(np.mean(not_use_rates)) 
            all_results += 'Avg Word Count: {:d}\n\n'.format(int(np.mean(word_counts)))
            
    with open(f'LLAMA3/{mode}/Results.txt', 'w') as f:
        f.write(all_results)

if __name__ == "__main__":
    main()
