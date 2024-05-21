import torch
import random
import os
from api import  efficient_openai_text_api
from utils import set_rand_seed
from load_graph import generate_grb_split
from LLM_utils import get_one_hop_neighbors, get_two_hop_neighbors, get_sampled_nodes, num_tokens_from_messages, load_mapping, few_shot
import numpy as np
import os.path as osp
import ast
import editdistance
import torch_geometric.transforms as T
import argparse


def prompt_zero_shot(data_obj, sampled_test_node_idxs, train_node_idxs, topk = True, need_class = True, instruction_format = 'arxiv cs xx', mapping = None, all_possible = False, cot = False, memory_limit = 1000000):
    if len(data_obj.raw_texts) < memory_limit:
        raw_texts = np.array(data_obj.raw_texts)
        nl_labels = np.array(data_obj.category_names)
    else:
        raw_texts = data_obj.raw_texts
        nl_labels = data_obj.category_names
    nu_labels = data_obj.y.numpy()
    label_names = data_obj.label_names
    if "arxiv" in instruction_format:
        label_names = [transform_category(x) for x in label_names]
    if mapping != None:
        human_label_names = [mapping[key] for key in data_obj.label_names]
    data_y = data_obj.y.numpy()
    prompts = []
    if len(data_obj.raw_texts) < memory_limit:
        selected_raw_texts = raw_texts[sampled_test_node_idxs]
        selected_y = data_y[sampled_test_node_idxs]
        selected_category = nl_labels[sampled_test_node_idxs]
    else:
        selected_raw_texts = [raw_texts[i] for i in sampled_test_node_idxs]
        selected_y = data_y[sampled_test_node_idxs]
        selected_category = [nl_labels[i] for i in sampled_test_node_idxs]
    for t in selected_raw_texts:
        if len(data_obj.raw_texts) < memory_limit:
            prompt = "Paper:\n {}\n".format(t)
        else:
            prompt = "Product Description:\n {}\n".format(t)
        if need_class:
            if mapping != None:
                prompt += "Task: \n"
                prompt += "There are following categories: \n"
                prompt += (str(human_label_names) + "\n")
            else:
                prompt += f"There are {nu_labels.max() + 1} classes:\n"
                prompt += (str(label_names) + "\n")
        if not need_class:
            prompt += "Which arxiv cs subcategories does this paper belong to?\n"
        else:
            if len(data_obj.raw_texts) < memory_limit:
                prompt += "Which category does this paper belong to?\n"
            else:
                prompt += "Which category does this product from Amazon belong to?\n"
        if topk:
            if instruction_format != "":
                if len(data_obj.raw_texts) < memory_limit:
                    prompt += f"Output the most 3 possible categories of this paper as a python list, with the format ['{instruction_format}', '{instruction_format}', '{instruction_format}']"
                else:
                    prompt += f"Output the most 3 possible categories of this product as a python list, with the format ['{instruction_format}', '{instruction_format}', '{instruction_format}']"
            else:
                if len(data_obj.raw_texts) < memory_limit:
                    prompt += f"Output the most 3 possible categories of this paper as a python list"
                else:
                    prompt += f"Output the most 3 possible categories of this product as a python list"
        elif all_possible:
            if len(data_obj.raw_texts) < memory_limit:
                prompt += f"Output all possible categories of this paper as a python list, like ['XX']"
            else:
                prompt += f"Output all possible categories of this product as a python list, like ['XX']"
        else:
            thought_process = "" if not cot else ". Think it step by step and output your reason in one sentence.\n"
            if instruction_format != "":
                if len(data_obj.raw_texts) < memory_limit:
                    prompt += f"Output the most 1 possible category of this paper as a python list, like ['{instruction_format}']{thought_process}"
                else:
                    prompt += f"Output the most 1 possible category of this product as a python list, like ['{instruction_format}']{thought_process}"
            else:
                if len(data_obj.raw_texts) < memory_limit:
                    prompt += f"Output the most 1 possible category of this paper as a python list"
                else:
                    prompt += f"Output the most 1 possible category of this product as a python list"
        prompts.append(prompt)
    if mapping != None:
        return prompts, selected_y, selected_category, human_label_names
    else:
        return prompts, selected_y, selected_category


def topk_accuracy(pred_texts, gt, label_names, topk = True, need_clean = True):
    preds = []
    correct = 0
    miss = 0
    label_names = [x.lower() for x in label_names]
    for i, t in enumerate(pred_texts):
        if need_clean:
            clean_t = t.replace('.', ' ')
            clean_t = clean_t.lower()
            clean_t = clean_t.replace('\\', '')
            clean_t = clean_t.replace('_', ' ')
        else:
            clean_t = t
        try:
            start = clean_t.find('[')
            end = clean_t.find(']', start) + 1  # +1 to include the closing bracket
            list_str = clean_t[start:end]
            result = ast.literal_eval(list_str)
            if res in label_names:
                this = label_names.index(res)
                if this == gt[i]:
                    correct += 1
                    continue
            else:
                miss += 1
                edits = np.array([editdistance.eval(res, l) for l in label_names])
                this = np.argmin(edits)
                if this == gt[i]:
                    correct += 1
                    continue
            
        except Exception:
            miss += 1
            for k, l in enumerate(label_names):
                if l.lower() in clean_t:
                    if k == gt[i]:
                        correct += 1
                    break
    print(miss)
    return correct / len(pred_texts)


def prompt_few_shot(data_obj, few_shot_samples, sampled_test_node_idxs, train_node_idxs, topk = True, need_class = True, instruction_format = 'arxiv cs xx', mapping = None, cot = False, dataset_name = "cora", shots = 3, memory_limit = 1000000):
    if len(data_obj.raw_texts) < memory_limit:
        raw_texts = np.array(data_obj.raw_texts)
        nl_labels = np.array(data_obj.category_names)
    else:
        raw_texts = data_obj.raw_texts
        nl_labels = data_obj.category_names
    nu_labels = data_obj.y.numpy()
    label_names = data_obj.label_names
    if mapping != None:
        human_label_names = [mapping[key] for key in data_obj.label_names]
    data_y = data_obj.y.numpy()
    prompts = []
    if len(data_obj.raw_texts) < memory_limit:
        selected_raw_texts = raw_texts[sampled_test_node_idxs]
        selected_y = data_y[sampled_test_node_idxs]
        selected_category = nl_labels[sampled_test_node_idxs]
    else:
        selected_raw_texts = [raw_texts[i] for i in sampled_test_node_idxs]
        selected_y = data_y[sampled_test_node_idxs]
        selected_category = [nl_labels[i] for i in sampled_test_node_idxs]
    
    few_shot = few_shot_samples["top1"][dataset_name]
    cot_str = few_shot_samples["cot"][dataset_name]
    for t in selected_raw_texts:
        if not cot:
            prompt = "\n".join(few_shot[:shots])
        else:
            start = []
            for i, demo in enumerate(few_shot[:shots]):
                result_end = demo.index("Result:\n") + len("Result:")
                temp = "{}\n{}{}".format(demo[:result_end], cot_str[i], demo[result_end:])
                start.append(temp)
            prompt = "\n".join(start)
        if len(data_obj.raw_texts) < memory_limit:
            prompt += "Paper:\n {}\n".format(t)
        else:
            prompt += "Product Description:\n {}\n".format(t)
        if need_class:
            if mapping != None:
                prompt += "Task: \n"
                prompt += "There are following categories: \n"
                prompt += (str(human_label_names) + "\n")
            else:
                prompt += f"There are {nu_labels.max() + 1} classes:\n"
                prompt += (str(label_names) + "\n")
        if not need_class:
            prompt += "Which arxiv cs subcategories does this paper belong to?\n"
        else:
            if len(data_obj.raw_texts) < memory_limit:
                prompt += "Which category does this paper belong to?\n"
            else:
                prompt += "Which category does this product from Amazon belong to?\n"
        if topk:
            if instruction_format != "":
                if len(data_obj.raw_texts) < memory_limit:
                    prompt += f"Output the most 3 possible categories of this paper as a python list, with the format ['{instruction_format}', '{instruction_format}', '{instruction_format}']"
                else:
                    prompt += f"Output the most 3 possible categories of this product as a python list, with the format ['{instruction_format}', '{instruction_format}', '{instruction_format}']"
            else:
                if len(data_obj.raw_texts) < memory_limit:
                    prompt += f"Output the most 3 possible categories of this paper as a python list"
                else:
                    prompt += f"Output the most 3 possible categories of this product as a python list"
        else:
            if instruction_format != "":
                if len(data_obj.raw_texts) < memory_limit:
                    prompt += f"Output the most 1 possible category of this paper as a python list, like ['{instruction_format}']"
                else:
                    prompt += f"Output the most 1 possible category of this product as a python list, like ['{instruction_format}']"
            else:
                if len(data_obj.raw_texts) < memory_limit:
                    prompt += f"Output the most 1 possible category of this paper as a python list"
                else:
                    prompt += f"Output the most 1 possible category of this product as a python list"
        prompt += "\nResult:"
        prompts.append(prompt)
    if mapping != None:
        return prompts, selected_y, selected_category, human_label_names
    else:
        return prompts, selected_y, selected_category


def generate_neighbor_information(data_obj, neighbors_dict, sampled_test_node_idxs, train_node_idxs, mapping = None, dataset_name = "cora", sample_num = 10, memory_limit = 1000000):
    if len(data_obj.raw_texts) < memory_limit:
        raw_texts = np.array(data_obj.raw_texts)
        nl_labels = np.array(data_obj.category_names)
    else:
        raw_texts = data_obj.raw_texts
        nl_labels = data_obj.category_names
    nl_labels = [mapping[x] for x in nl_labels]
    nu_labels = data_obj.y.numpy()
    label_names = data_obj.label_names
    if mapping != None:
        human_label_names = [mapping[key] for key in data_obj.label_names]
    data_y = data_obj.y.numpy()
    prompts = []    
    typ = "products" if dataset_name == 'products' else "papers"
    for idx in sampled_test_node_idxs:
        prompt = f"The following list records some {typ} related to the current one."
        sam_num = min(sample_num, len(neighbors_dict[idx.item()]))
        sampled_nei = random.sample(neighbors_dict[idx.item()], sam_num)
        this_neighbors = sampled_nei
        contexts = []
        total_num_of_tokens = 0
        for nei in this_neighbors:
            texts = data_obj.raw_texts[nei]
            if nei in train_node_idxs:
                category = nl_labels[nei]
                this_context = {
                    "content": texts,
                    "category": category
                }
            else:
                this_context = {
                    "content": texts
                }
            this_context_str = str(this_context)
            this_num_token = num_tokens_from_messages(this_context_str)
            if total_num_of_tokens + this_num_token > 3000:
                break 
            total_num_of_tokens += this_num_token
            contexts.append(this_context)
        prompt += str(contexts)
        prompt += "\nPlease summarize the information above with a short paragraph, find some common points which can reflect the category of this paper\n"
        prompts.append(prompt)
    return prompts


def generate_demo(data_obj, few_shot_texts, human_label_names, nu_labels, label_names, few_shot_labels, dataset_name = "products", summary = [], instruction_format = 'XX', mapping = None, need_class = False, memory_limit = 1000000):
    prompt = ""
    for i, s in enumerate(summary[:3]):
        if dataset_name != 'products':
            prompt += "Paper:\n {}\n".format(few_shot_texts[i])
        else:
            prompt += "Product Description:\n {}\n".format(few_shot_texts[i])
        prompt += "Neighbor Summary: \n {} \n".format(s)
        if need_class:
            if mapping != None:
                prompt += "Task: \n"
                prompt += "There are following categories: \n"
                prompt += (str(human_label_names) + "\n")
            else:
                prompt += f"There are {nu_labels.max() + 1} classes:\n"
                prompt += (str(label_names) + "\n")
        if not need_class:
            prompt += "Which arxiv cs subcategories does this paper belong to?\n"
        else:
            if len(data_obj.raw_texts) < memory_limit:
                prompt += "Which category does this paper belong to?\n"
            else:
                prompt += "Which category does this product from Amazon belong to?\n"
        if instruction_format != "":
            if len(data_obj.raw_texts) < memory_limit:
                prompt += f"Output the most 1 possible category of this paper as a python list, like ['{instruction_format}']"
            else:
                prompt += f"Output the most 1 possible category of this product as a python list, like ['{instruction_format}']"
        else:
            if len(data_obj.raw_texts) < memory_limit:
                prompt += f"Output the most 1 possible category of this paper as a python list"
            else:
                prompt += f"Output the most 1 possible category of this product as a python list"
        prompt += "\nResult:\n[\"{}\"]\n".format(few_shot_labels[i])
    return prompt


def prompt_with_neighbor_summary(data_obj, few_shot_texts, neighbor_summary, few_shot_demos, few_shot_labels, sampled_test_node_idxs, train_node_idxs, topk = True, need_class = True, instruction_format = 'arxiv cs xx', mapping = None, dataset_name = "cora", shots = 3, memory_limit = 1000000):
    if len(data_obj.raw_texts) < memory_limit:
        raw_texts = np.array(data_obj.raw_texts)
        nl_labels = np.array(data_obj.category_names)
    else:
        raw_texts = data_obj.raw_texts
        nl_labels = data_obj.category_names
    nu_labels = data_obj.y.numpy()
    label_names = data_obj.label_names
    if mapping != None:
        human_label_names = [mapping[key] for key in data_obj.label_names]
    else:
        human_label_names = None
    data_y = data_obj.y.numpy()
    prompts = []
    if len(data_obj.raw_texts) < memory_limit:
        selected_raw_texts = raw_texts[sampled_test_node_idxs]
        selected_y = data_y[sampled_test_node_idxs]
        selected_category = nl_labels[sampled_test_node_idxs]
    else:
        selected_raw_texts = [raw_texts[i] for i in sampled_test_node_idxs]
        selected_y = data_y[sampled_test_node_idxs]
        selected_category = [nl_labels[i] for i in sampled_test_node_idxs]
    
    for i, t in enumerate(selected_raw_texts):
        summary = neighbor_summary[i]
        ## zero shot
        prompt = ""
        if shots != 0:
            prompt += generate_demo(data_obj, few_shot_texts, human_label_names, nu_labels, label_names, few_shot_labels, dataset_name = dataset_name, summary = few_shot_demos, instruction_format = instruction_format, mapping = mapping, need_class=need_class)
        if len(data_obj.raw_texts) < memory_limit:
            prompt += "Paper:\n {}\n".format(t)
        else:
            prompt += "Product Description:\n {}\n".format(t)
        prompt += "Neighbor Summary: \n {} \n".format(summary)
        if need_class:
            if mapping != None:
                prompt += "Task: \n"
                prompt += "There are following categories: \n"
                prompt += (str(human_label_names) + "\n")
            else:
                prompt += f"There are {nu_labels.max() + 1} classes:\n"
                prompt += (str(label_names) + "\n")
        if not need_class:
            prompt += "Which arxiv cs subcategories does this paper belong to?\n"
        else:
            if len(data_obj.raw_texts) < memory_limit:
                prompt += "Which category does this paper belong to?\n"
            else:
                prompt += "Which category does this product from Amazon belong to?\n"
        if topk:
            if instruction_format != "":
                if len(data_obj.raw_texts) < memory_limit:
                    prompt += f"Output the most 3 possible categories of this paper as a python list, with the format ['{instruction_format}', '{instruction_format}', '{instruction_format}']"
                else:
                    prompt += f"Output the most 3 possible categories of this product as a python list, with the format ['{instruction_format}', '{instruction_format}', '{instruction_format}']"
            else:
                if len(data_obj.raw_texts) < memory_limit:
                    prompt += f"Output the most 3 possible categories of this paper as a python list"
                else:
                    prompt += f"Output the most 3 possible categories of this product as a python list"
        else:
            if instruction_format != "":
                if len(data_obj.raw_texts) < memory_limit:
                    prompt += f"Output the most 1 possible category of this paper as a python list, like ['{instruction_format}']"
                else:
                    prompt += f"Output the most 1 possible category of this product as a python list, like ['{instruction_format}']"
            else:
                if len(data_obj.raw_texts) < memory_limit:
                    prompt += f"Output the most 1 possible category of this paper as a python list"
                else:
                    prompt += f"Output the most 1 possible category of this product as a python list"
        if shots > 0:
            prompt += "\nResult:\n"
        prompts.append(prompt)
    if mapping != None:
        return prompts, selected_y, selected_category, human_label_names
    else:
        return prompts, selected_y, selected_category

    
def transform_category(category):
    parts = category.split()
    if len(parts) != 3 or parts[0].lower() != 'arxiv' or parts[1].lower() != 'cs':
        raise ValueError("Input should be in the format 'arxiv cs xx'")
    return "{} {}.{}".format(parts[0], parts[1], parts[2].upper())


def print_to_file(lists, output_name = "abc.txt"):
    with open(output_name, "w") as f:
        for line in lists:
            f.write(line.replace('\n', ''))
            f.write("\n")


class ComprehensiveStudy:
    def __init__(self, dataset, file='', openai_dir='Graph-LLM', sample_num=200, eval_no_neighbor=False):
        self.datasets = [dataset]
        self.arxiv_mapping, self.citeseer_mapping, self.pubmed_mapping, self.cora_mapping, self.products_mapping = load_mapping()
        self.seeds = [0]
        self.sample_num = sample_num
        self.file = file
        self.openai_dir = openai_dir
        os.makedirs(f'openai_io/{openai_dir}', exist_ok=True)
        self.eval_no_neighbor = eval_no_neighbor

    def prepare_dataset(self, dataset_name, seed):
        set_rand_seed(self.seeds[0])
        sample_num = self.sample_num
        if self.file == '':
            data = torch.load(f"./data/{dataset_name}_fixed_tfidf.pt", map_location = 'cpu')
            edge_index = data.edge_index
            data = T.ToSparseTensor()(data)
            data.edge_index = edge_index
            train_mask, val_mask, test_mask = generate_grb_split(data, mode='full')
            split_idx = {'train': torch.nonzero(train_mask, as_tuple=True)[0],
                    'valid':torch.nonzero(val_mask, as_tuple=True)[0], 
                    'test': torch.nonzero(test_mask, as_tuple=True)[0]}
            data.y = data.y.unsqueeze(1)
            data.train_mask, data.val_mask, data.test_mask = train_mask, val_mask, test_mask
        else:
            data = torch.load(self.file)
            raw_texts_len = len(data.raw_texts)
            n_nodes = data.x.shape[0]

            if raw_texts_len < n_nodes:
                for i in range(raw_texts_len, n_nodes):
                    data.raw_texts.append('')
                
        print(data)

        set_rand_seed(self.seeds[0])
        if osp.exists(f"./LLMGNN_output/selected_{dataset_name}_{seed}_{sample_num}.pt"):
            selected_pt = torch.load(f"./LLMGNN_output/selected_{dataset_name}_{seed}_{sample_num}.pt", map_location="cpu")
            sampled_test_node_idxs, train_node_idxs = selected_pt
        else:
            sampled_test_node_idxs, train_node_idxs = get_sampled_nodes(data, sample_num)
            torch.save((sampled_test_node_idxs, train_node_idxs),f"./LLMGNN_output/selected_{dataset_name}_{seed}_{sample_num}.pt")

        one_hop_neighbors = get_one_hop_neighbors(data, sampled_test_node_idxs, sample_num)
        two_hop_neighbors = get_two_hop_neighbors(data, sampled_test_node_idxs, sample_num)
        
        print(f"{dataset_name} data processed!")
        if dataset_name == "arxiv":
            mapping = self.arxiv_mapping
        elif dataset_name == 'citeseer':
            mapping = self.citeseer_mapping
        elif dataset_name == 'pubmed':
            mapping = self.pubmed_mapping
        elif dataset_name == 'cora':
            mapping = self.cora_mapping

        return mapping, data, sampled_test_node_idxs, train_node_idxs, one_hop_neighbors, two_hop_neighbors


    def zero_shot(self, dataset, sampled_test_node_idxs, train_node_idxs, mapping, seed, dataset_name, instruction='XX'):
        zero_shot_prompt_human_top1, y, cat_names, human_labels = prompt_zero_shot(dataset, sampled_test_node_idxs, train_node_idxs, topk=False, need_class=True, instruction_format=instruction, mapping = mapping)
        input_file_name = f"./openai_io/{self.openai_dir}/{dataset_name}_human_zero_shot_top1_classification_input_{seed}.json"
        output_file_name = f"./openai_io/{self.openai_dir}/{dataset_name}_human_zero_shot_top1_classification_output_{seed}.json"
        results = efficient_openai_text_api(zero_shot_prompt_human_top1, input_file_name, output_file_name, sp=60, ss=1.5, rewrite=False)
        human_zero_shot_top1_pred_texts = [x[0] for x in results]
        print_to_file(cat_names, f"./openai_io/{self.openai_dir}/{dataset_name}_human_zero_shot_top1_classification_labels_{seed}.txt")
        print_to_file(human_zero_shot_top1_pred_texts, f"./openai_io/{self.openai_dir}/{dataset_name}_human_zero_shot_top1_classification_output_{seed}.txt")
        top1_acc = topk_accuracy(human_zero_shot_top1_pred_texts, y, human_labels, topk = False)
        print(f"{dataset_name} human zero shot top1 acc: {top1_acc}")
    
    def few_shot(self, dataset, sampled_test_node_idxs, train_node_idxs, mapping, seed, dataset_name, instruction='XX'):
        few_shot_samples = few_shot()
        few_shot_prompt_human_top1, y, cat_names, human_labels = prompt_few_shot(dataset, few_shot_samples, sampled_test_node_idxs, train_node_idxs, topk = False, need_class = True, instruction_format = instruction, mapping = mapping, cot = False, dataset_name = dataset_name, shots = 3)
        input_file_name = f"./openai_io/{self.openai_dir}/{dataset_name}_human_few_shot_top1_classification_input_{seed}.json"
        output_file_name = f"./openai_io/{self.openai_dir}/{dataset_name}_human_few_shot_top1_classification_output_{seed}.json"
        results = efficient_openai_text_api(few_shot_prompt_human_top1, input_file_name, output_file_name, sp=60, ss=1.5, rewrite=False)
        human_few_shot_top1_pred_texts = [x[0] for x in results]
        print_to_file(human_few_shot_top1_pred_texts, f"./openai_io/{self.openai_dir}/{dataset_name}_human_few_shot_top1_classification_output_{seed}.txt")
        top1_acc = topk_accuracy(human_few_shot_top1_pred_texts, y, human_labels, topk = False)
        print(f"{dataset_name} human few shot top1 acc: {top1_acc}")
        return few_shot_samples
    
    def get_nei_info(self, dataset, two_hop_neighbors, sampled_test_node_idxs, train_node_idxs, mapping, seed, dataset_name):
        two_hop_info = generate_neighbor_information(dataset, two_hop_neighbors, sampled_test_node_idxs, train_node_idxs, mapping=mapping, dataset_name = dataset_name, sample_num = 10)
        input_file_name = f"./openai_io/{self.openai_dir}/new_{dataset_name}_2hop_neighbors_input_{seed}.json"
        output_file_name = f"./openai_io/{self.openai_dir}/new_{dataset_name}_2hop_neighbors_output_{seed}.json"
        results = efficient_openai_text_api(two_hop_info, input_file_name, output_file_name, sp=60, ss=1.5, rewrite=False)
        two_hop_summary = [x[0] for x in results]
        return two_hop_summary
    
    def zero_shot_with_2hop_nei_info(self, dataset, two_hop_summary, sampled_test_node_idxs, train_node_idxs, mapping, seed, dataset_name):
        zero_shot_two_hop_prompt_top1, y, cat_names, human_labels = prompt_with_neighbor_summary(dataset, None, two_hop_summary, None, None, sampled_test_node_idxs, train_node_idxs, False, True, 'XX', mapping=mapping, dataset_name=dataset_name, shots=0)
        input_file_name = f"./openai_io/{self.openai_dir}/new_{dataset_name}_human_2_hop_zero_shot_top1_classification_input_{seed}.json"
        output_file_name = f"./openai_io/{self.openai_dir}/new_{dataset_name}_human_2_hop_zero_shot_top1_classification_output_{seed}.json"
        results = efficient_openai_text_api(zero_shot_two_hop_prompt_top1, input_file_name, output_file_name, sp=60, ss=1.5, rewrite=False)
        zero_shot_two_hop_top1_pred_texts = [x[0] for x in results]
        print_to_file(zero_shot_two_hop_top1_pred_texts, f"./openai_io/{self.openai_dir}/{dataset_name}_human_2_hop_zero_shot_top1_classification_output_{seed}.txt")
        top1_acc = topk_accuracy(zero_shot_two_hop_top1_pred_texts, y, human_labels, topk = False)
        print(f"{dataset_name} human zero shot 2hop top1 acc: {top1_acc}")

    def get_few_shot_with_2hop_nei_info(self, dataset, two_hop_summary, sampled_test_node_idxs, train_node_idxs, mapping, seed, dataset_name):
        few_shot_samples = torch.IntTensor(random.sample(train_node_idxs.tolist(), 3))
        few_shot_labels = [dataset.category_names[i.item()] for i in few_shot_samples]
        few_shot_labels = [mapping[x] for x in few_shot_labels]
        few_shot_texts = [dataset.raw_texts[i.item()] for i in few_shot_samples] 
        sample_num = self.sample_num
        sample_two_hop_neighbors = get_two_hop_neighbors(dataset, few_shot_samples, sample_num)
        two_hop_info = generate_neighbor_information(dataset, sample_two_hop_neighbors, few_shot_samples, train_node_idxs, mapping=mapping, dataset_name = dataset_name)
        input_file_name = f"./openai_io/{self.openai_dir}/new_{dataset_name}_sample_2hop_neighbors_input_{seed}.json"
        output_file_name = f"./openai_io/{self.openai_dir}/new_{dataset_name}_sample_2hop_neighbors_output_{seed}.json"
        results = efficient_openai_text_api(two_hop_info, input_file_name, output_file_name, sp=60, ss=1.5, rewrite=False)
        few_shot_two_hop_summary = [x[0] for x in results]
        return few_shot_two_hop_summary, few_shot_labels, few_shot_texts
    
    def few_shot_with_2hop_nei_info(self, dataset, few_shot_texts, two_hop_summary, few_shot_two_hop_summary, few_shot_labels, sampled_test_node_idxs, train_node_idxs, mapping, seed, dataset_name, instruction='XX'):
        few_shot_2hop_prompt, y, cat_names, human_labels = prompt_with_neighbor_summary(dataset, few_shot_texts, two_hop_summary, few_shot_two_hop_summary, few_shot_labels, \
            sampled_test_node_idxs, train_node_idxs, topk = False, need_class=True, instruction_format=instruction, mapping=mapping, dataset_name = dataset_name)
        # import ipdb; ipdb.set_trace()
        input_file_name = f"./openai_io/{self.openai_dir}/new_{dataset_name}_human_few_shot_2hop_top1_classification_input_new_{seed}.json"
        output_file_name = f"./openai_io/{self.openai_dir}/new_{dataset_name}_human_few_shot_2hop_top1_classification_output_new_{seed}.json"
        results = efficient_openai_text_api(few_shot_2hop_prompt, input_file_name, output_file_name, sp=60, ss=1.5, rewrite=False)
        human_few_shot_2hop_top1_pred_texts = [x[0] for x in results]
        print_to_file(human_few_shot_2hop_top1_pred_texts, f"./openai_io/{self.openai_dir}/{dataset_name}_human_few_shot_2hop_top1_classification_output_{seed}.txt")
        top1_acc = topk_accuracy(human_few_shot_2hop_top1_pred_texts, y, human_labels, topk = False)
        print(f"{dataset_name} human few shot 2hop top1 acc: {top1_acc}")


    def full_run(self):
        for seed in self.seeds:
            for dataset_name in self.datasets:
                mapping, dataset, sampled_test_node_idxs, train_node_idxs, one_hop_neighbors, two_hop_neighbors = self.prepare_dataset(dataset_name, seed)
                if self.eval_no_neighbor:
                    print("Zero shot")
                    self.zero_shot(dataset, sampled_test_node_idxs, train_node_idxs, mapping, seed, dataset_name)
                    print("Few shot")
                    few_shot_samples = self.few_shot(dataset, sampled_test_node_idxs, train_node_idxs, mapping, seed, dataset_name)
                two_hop_summary = self.get_nei_info(dataset, two_hop_neighbors, sampled_test_node_idxs, train_node_idxs, mapping, seed, dataset_name)
                few_shot_two_hop_summary, few_shot_labels, few_shot_texts = self.get_few_shot_with_2hop_nei_info(dataset, two_hop_neighbors, sampled_test_node_idxs, train_node_idxs, mapping, seed, dataset_name)
                print("Zero shot with 2 hop neighborhood")
                self.zero_shot_with_2hop_nei_info(dataset, two_hop_summary, sampled_test_node_idxs, train_node_idxs, mapping, seed, dataset_name)
                print("Few shot with 2 hop neighborhood")
                self.few_shot_with_2hop_nei_info(dataset, few_shot_texts, two_hop_summary, few_shot_two_hop_summary, few_shot_labels, sampled_test_node_idxs, train_node_idxs, mapping, seed, dataset_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default='')
    parser.add_argument('--save_dir', type=str, default='Graph-LLM/cora_llmgia')
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--sample_num', type=int, default=200)
    parser.add_argument('--eval_no_neighbor', action='store_true')
    args = parser.parse_args()
    #os.makedirs(args.save_dir, exist_ok=True)
    study = ComprehensiveStudy(dataset=args.dataset, file=args.file, openai_dir=args.save_dir, sample_num=args.sample_num, eval_no_neighbor=args.eval_no_neighbor)
    study.full_run()


if __name__ == '__main__':
    main()