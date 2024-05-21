import sys
sys.path.append("../")

import torch
import torch.nn.functional as F
import numpy as np
import os
import pickle
from api import efficient_openai_text_api
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sentence_transformers import SentenceTransformer
import random


def save_vectorizer(vec, filename):
    with open(filename, 'wb') as f:
        pickle.dump(vec, f)


def load_vectorizer(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def get_bow_by_texts(texts, dataset, max_features=500, known_mask=None):
    vectorizer_path = os.path.join("./bow_cache/", f"{dataset}.pkl")
    if known_mask is None:
        if vectorizer_path and os.path.exists(vectorizer_path):
            print("Loading bow .pkl")
            vec = load_vectorizer(vectorizer_path)
            X = vec.transform(texts)
        else:
            vec = CountVectorizer(max_features=max_features, stop_words='english', binary=True)
            X = vec.fit_transform(texts)
            if vectorizer_path:
                save_vectorizer(vec, vectorizer_path) 
        torch_feat = torch.FloatTensor(X.toarray())
        norm_torch_feat = F.normalize(torch_feat, p=2, dim=-1)
        return torch_feat, norm_torch_feat
    else:
        if vectorizer_path and os.path.exists(vectorizer_path):
            print("Loading bow .pkl")
            vec = load_vectorizer(vectorizer_path)
        else:
            print("Saving bow .pkl")
            vec = CountVectorizer(max_features=max_features, stop_words='english', binary=True)
            texts_known = texts[known_mask]
            vec.fit(texts_known)
            if vectorizer_path:
                save_vectorizer(vec, vectorizer_path)

        x_known = vec.transform(texts[known_mask])
        x_test = vec.transform(texts[~known_mask])
        x_known = torch.FloatTensor(x_known.todense())
        x_test = torch.FloatTensor(x_test.todense())
        dim = x_known.shape[1]
        torch_feat = torch.zeros(len(texts), dim)
        torch_feat[known_mask] = x_known
        torch_feat[~known_mask] = x_test
        norm_torch_feat = F.normalize(torch_feat, dim=-1)

        return torch_feat, norm_torch_feat


def get_tf_idf_by_texts(texts, max_features=500, use_tokenizer=False):
    tf_idf_vec = TfidfVectorizer(stop_words='english', max_features=max_features)
    X = tf_idf_vec.fit_transform(texts)
    torch_feat = torch.FloatTensor(X.todense())
    norm_torch_feat = F.normalize(torch_feat, dim = -1)
    return torch_feat, norm_torch_feat
    

def get_sbert_emb(texts, device):
    model = SentenceTransformer(cache_folder='./cache', device=device).to(device)
    sbert_embeds = model.encode(texts, batch_size=8, show_progress_bar=True)
    feat = torch.tensor(sbert_embeds)
    return feat


def get_gtr_emb(texts, batch_size=32) -> torch.Tensor:
    model = SentenceTransformer('sentence-transformers/gtr-t5-base')

    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_embeddings = model.encode(batch_texts, convert_to_tensor=True,).to('cuda')
        embeddings.append(batch_embeddings)
    embeddings = torch.cat(embeddings, dim=0)

    return embeddings


def text2emb(texts, dataset, embdding='bow', known_mask=None):
    if embdding == 'bow':
        x, norm_x = get_bow_by_texts(texts, dataset, known_mask=known_mask)
    elif embdding == 'sbert':
        x = get_sbert_emb(texts, device="cuda")
    elif embdding == 'tfidf':
        x, norm_x = get_tf_idf_by_texts(texts, known_mask=known_mask)
    elif embdding == 'gtr':
        x = get_gtr_emb(texts)
    return x


def text_stats(texts):
    word_counts = [len(text.split()) for text in texts]
    max_words = max(word_counts)
    min_words = min(word_counts)
    average_words = np.mean(word_counts)
    std_deviation = np.std(word_counts)
    
    return min_words, max_words, average_words, std_deviation


def clear_text(raw_text):
    # Remove "title"/"abs"/"\n"
    raw_text = raw_text.replace("\n", "")
    raw_text = raw_text.replace("Title: ", "")
    raw_text = raw_text.replace("Abstract:", "")
    raw_text = raw_text.replace("\"", "")
    return raw_text



def raw_data_to_dataset_name(features):
    num_nodes = features.shape[0]
    if num_nodes == 2708:
        dataset_name = 'cora'
    elif num_nodes == 3186:
        dataset_name = 'citeseer'
    elif num_nodes == 19717:
        dataset_name = 'pubmed'
    else:
        dataset_name = 'arxiv'
    return dataset_name


def load_mapping(dataset=None):
    arxiv_mapping = {'arxiv cs ai': 'Artificial Intelligence', 'arxiv cs cl': 'Computation and Language', 'arxiv cs cc': 'Computational Complexity', 'arxiv cs ce': 'Computational Engineering, Finance, and Science', 'arxiv cs cg': 'Computational Geometry', 'arxiv cs gt': 'Computer Science and Game Theory', 'arxiv cs cv': 'Computer Vision and Pattern Recognition', 'arxiv cs cy': 'Computers and Society', 'arxiv cs cr': 'Cryptography and Security', 'arxiv cs ds': 'Data Structures and Algorithms', 'arxiv cs db': 'Databases', 'arxiv cs dl': 'Digital Libraries', 'arxiv cs dm': 'Discrete Mathematics', 'arxiv cs dc': 'Distributed, Parallel, and Cluster Computing', 'arxiv cs et': 'Emerging Technologies', 'arxiv cs fl': 'Formal Languages and Automata Theory', 'arxiv cs gl': 'General Literature', 'arxiv cs gr': 'Graphics', 'arxiv cs ar': 'Hardware Architecture', 'arxiv cs hc': 'Human-Computer Interaction', 'arxiv cs ir': 'Information Retrieval', 'arxiv cs it': 'Information Theory', 'arxiv cs lo': 'Logic in Computer Science', 'arxiv cs lg': 'Machine Learning', 'arxiv cs ms': 'Mathematical Software', 'arxiv cs ma': 'Multiagent Systems', 'arxiv cs mm': 'Multimedia', 'arxiv cs ni': 'Networking and Internet Architecture', 'arxiv cs ne': 'Neural and Evolutionary Computing', 'arxiv cs na': 'Numerical Analysis', 'arxiv cs os': 'Operating Systems', 'arxiv cs oh': 'Other Computer Science', 'arxiv cs pf': 'Performance', 'arxiv cs pl': 'Programming Languages', 'arxiv cs ro': 'Robotics', 'arxiv cs si': 'Social and Information Networks', 'arxiv cs se': 'Software Engineering', 'arxiv cs sd': 'Sound', 'arxiv cs sc': 'Symbolic Computation', 'arxiv cs sy': 'Systems and Control'}
    citeseer_mapping = {
        "Agents": "Agents",
        "ML": "Machine Learning",
        "IR": "Information Retrieval",
        "DB": "Database",
        "HCI": "Human Computer Interaction",
        "AI": "Artificial Intelligence"
    }
    pubmed_mapping = {
        'Diabetes Mellitus, Experimental': 'Diabetes Mellitus, Experimental',
        'Diabetes Mellitus Type 1': 'Diabetes Mellitus Type 1',
        'Diabetes Mellitus Type 2': 'Diabetes Mellitus Type 2'
    }
    cora_mapping = {
        'Rule_Learning': "Rule Learning",
        'Neural_Networks': "Neural Networks",
        'Case_Based': "Case Based",
        'Genetic_Algorithms': "Genetic Algorithms",
        'Theory': "Theory",
        'Reinforcement_Learning': "Reinforcement Learning",
        'Probabilistic_Methods': "Probabilistic Methods"
    }

    if dataset is None:
        return arxiv_mapping, citeseer_mapping, pubmed_mapping, cora_mapping
    elif dataset.lower() == 'cora':
        return cora_mapping
    elif dataset.lower() == 'citeseer':
        return citeseer_mapping
    elif dataset.lower() == 'pubmed':
        return pubmed_mapping
    elif dataset.lower() == 'arxiv':
        return arxiv_mapping



def example_nodes_mask(raw_data, pred_orig, prob_threshold = 0.99, min_text_length = 50, max_text_length = 1000):
    labeled_mask = raw_data.train_mask | raw_data.val_mask
    predicted_labels = torch.argmax(pred_orig, dim=1)
    correct_combined = (predicted_labels == raw_data.y.view(-1)) & labeled_mask
    max_prob = torch.zeros_like(raw_data.y.view(-1), dtype=torch.float32)
    max_prob[correct_combined] = pred_orig[correct_combined].max(dim=1).values
    high_prob_mask = correct_combined & (max_prob > prob_threshold)
    
    text_length_mask = torch.tensor([max_text_length >= len(raw_data.raw_texts[node].split()) >= min_text_length \
                                     for node in range(len(raw_data.raw_texts))]).to(high_prob_mask.device)
    final_selected_mask = high_prob_mask & text_length_mask
    example_node_mask = final_selected_mask
    return example_node_mask


def example_node_sampler(raw_data, pred_orig, seed, num_examples=5, target_label=-1):
    raw_texts = raw_data.raw_texts
    min_len, max_len, avg_len, std_len = text_stats(raw_texts)
    example_mask = example_nodes_mask(raw_data, pred_orig, min_text_length=avg_len - std_len, max_text_length=avg_len + std_len)
    if target_label != -1:
        matching_indices = ((raw_data.y[raw_data.train_mask | raw_data.val_mask] == target_label) & example_mask).nonzero(as_tuple=True)[0]
    else:
        matching_indices = example_mask.nonzero(as_tuple=True)[0]
    local_rng = torch.Generator().manual_seed(seed)
    selected_indices = matching_indices[torch.randperm(len(matching_indices), generator=local_rng)[:num_examples]].tolist()
    labels = [raw_data.label_names[raw_data.y[i]] for i in selected_indices]
    return selected_indices, labels, [raw_texts[i] for i in selected_indices]



def text_feature_init(raw_data, pred_orig, n_inject_nodes, prompt_type):
    prompt_list = [] # store the prompt for each inject node
    raw_texts = raw_data.raw_texts
    label_names = raw_data.label_names

    num_nodes = raw_data.x.shape[0]
    dataset_name = raw_data_to_dataset_name(raw_data.x)
    min_len, max_len, avg_len, std_len = text_stats(raw_texts)
    mapping = load_mapping(dataset_name)
    label_names = [mapping[label] for label in label_names]
    num_classes = len(label_names)

    if prompt_type == 'sample':
        atk_data = torch.load(f"./atkg/gtr/{dataset_name}_atdgia_0.pt") # Reference atk file
        atk_edge_index = atk_data.edge_index
        atk_edge = atk_edge_index[:, atk_edge_index[0] >= num_nodes]
    

    for inj_node in range(n_inject_nodes):
        if prompt_type == 'sample':
        # Sample positive and negative examples
            # Retrieve attack edge 
            edge_dict = {}
            raw_text_dict = {}
            for col in range(atk_edge.size(1)):
                inj_node = atk_edge[0, col].item()
                ori_node = atk_edge[1, col].item()
                if ori_node < num_nodes: # TODO: neglect inj-inj edges
                    raw_text = raw_data.raw_texts[ori_node]
                else:
                    raw_text = ""
                if inj_node in edge_dict:
                    edge_dict[inj_node].append(ori_node)
                    raw_text_dict[inj_node].append(raw_text)
                else:
                    edge_dict[inj_node] = [ori_node]
                    raw_text_dict[inj_node] = [raw_text]

            # Sampling
            idx, emp_labels, emp_texts = example_node_sampler(raw_data, pred_orig, seed=inj_node*123, num_examples=3)
            prompt = f"Task: Paper Generation. \n There are {num_classes} types of paper, which are {label_names}. \n"
            prompt += f"Positive Examples of the papers are: \n"
            for i in range(len(idx)):
                prompt += f"Content: {emp_texts[i]} Type:{emp_labels[i]}\n"
            
            prompt += f"Negative Examples of the papers are: \n"
                
            adj_paper_content = ""
            adj_node_list = random.sample(raw_text_dict[inj_node], 3) # The list of raw text of adjacent nodes of inj_node
            for i, paper in enumerate(adj_node_list):
                adj_paper_content = f"Content: {paper}. \n"
                prompt += adj_paper_content
            
            prompt += f"Generate a title and an abstract for paper {inj_node + num_nodes} which is dissimilar to the negative examples, but belongs to at least one of the types in {label_names} similar to positive examples.\n"
            prompt += f"Length limit: Min: {int(min_len)} words, Max: {int(max_len)} words.\n"
            prompt += f"Title: ..., Abstract: ..."
            prompt_list.append(prompt)
        elif prompt_type == 'mixing': 
        # Mixing paper types
            prompt = f"Task: Paper Generation. \n There are {num_classes} types of paper, which are {label_names}. "
            examples = [example_node_sampler(raw_data, pred_orig, seed=inj_node*234, num_examples=1, target_label=i) for i in range(num_classes)]
            prompt += f"Examples of the papers are: \n"
            for i, emp_tuple in enumerate(examples):
                idx, emp_label, emp_text = emp_tuple
                prompt += f"Content: {emp_text} Type:{emp_label}\n"
            prompt += f"Generate a title and an abstract for paper {inj_node + num_nodes} that belong to all the above paper types. \n"
            prompt += f"The generate content should be able to be classifed into any types of paper."
            prompt += f"Length limit: Min: {int(min_len)} words, Max: {int(max_len)} words.\n"
            prompt += f"Title: ..., Abstract: ..."
            prompt_list.append(prompt)
        elif prompt_type == 'random':
            prompt = f"Task: Paper Generation. \n"
            prompt += f"Generate a title and an abstract for the paper {inj_node + num_nodes} belonging to ['history', 'art', 'philosophy', 'sports', 'music'].\n"
            prompt += f"Length limit: Min: {int(min_len)} words, Max: {int(max_len)} words.\n"
            prompt += f"Title: ..., Abstract: ..."
            prompt_list.append(prompt)
        else:
            raise NotImplementedError

    directory_path = f"./openai_io/{prompt_type}/{dataset_name}_tga_0"
    os.makedirs(directory_path, exist_ok=True)
    input_file_name = f"./openai_io/{prompt_type}/{dataset_name}_tga_0/input.json"
    output_file_name = f"./openai_io/{prompt_type}/{dataset_name}_tga_0/output.json"
    results = efficient_openai_text_api(prompt_list, input_file_name, output_file_name, sp=60, ss=1.5, rewrite=False)
    for raw_text_tuple in results:
        raw_text = raw_text_tuple[0]
        if dataset_name != 'pubmed':
            raw_text = clear_text(raw_text)
        raw_text = clear_text(raw_text)
        raw_texts.append(raw_text)
    
    return raw_texts