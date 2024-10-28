import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import string 
import random
import json
import torch 
import editdistance
import os.path as osp
from torch_geometric.utils import to_networkx, coalesce, to_torch_csr_tensor, to_edge_index, remove_self_loops
from tqdm import tqdm
import seaborn as sns
from pathlib import Path
import itertools
import tiktoken


def label_getter(result, label_names):
    l = []
    for line in result:
        find = False 
        for i, label in enumerate(label_names):
            if label.lower() in line:
                l.append(i)
                find = True
                break 
        if not find:
            edits = np.array([editdistance.eval(line, l) for l in label_names])
            l.append(np.argmin(edits))
    return torch.tensor(l)
    


def tsne_2d_plot(embeddings, filename, labels):
    # Apply t-SNE to project embeddings to 2D
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    for i, (x, y) in enumerate(embeddings_2d):
        plt.scatter(x, y)
        plt.text(x+0.01, y+0.01, labels[i], fontsize=9)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.title("2D t-SNE Visualization of Embeddings")
    plt.show()
    plt.savefig(filename)


# Function to plot tensors using t-SNE with different colors for different labels
def plot_tensors_with_tsne(tensors, labels):
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(tensors)

    plt.figure(figsize=(8, 6))
    for i, label in enumerate(np.unique(labels)):
        plt.scatter(tsne_results[labels == label, 0], tsne_results[labels == label, 1], label=label)

    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend()
    plt.title("2D t-SNE Visualization of Tensors with Different Colors for Labels")
    plt.show()




def random_string(length):
    # Choose from all possible characters (letters and digits)
    possible_chars = string.ascii_letters + string.digits
    
    # Generate a random string of the specified length
    result = ''.join(random.choice(possible_chars) for _ in range(length))
    
    return result


def read_jsonl(file_path):
    result = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line.strip())
            result.append(json_obj)
    return result


def error_analysis(output_file_path, ground_truth, label_names, gnn_output, mlp_output):
    result = read_jsonl(output_file_path)
    pred_texts = [x[1]['choices'][0]['message']['content'] for x in result]
    prompts = [x[0]['messages'][0]['content'] for x in result]
    preds = label_getter(pred_texts, label_names).numpy()
    gtnp = ground_truth.numpy()
    idx = np.arange(len(pred_texts))
    dataset = output_file_path.split('/')[-1].split("_")[0]
    dataset_path = Path(osp.join("./error_analysis", dataset))
    dataset_path.mkdir(parents=True, exist_ok=True)
    ## gpt is correct, mlp is correct
    mlpconfusion_matrix, mlpul, mlpur, mlpll, mlplr = create_confusion_matrix(gtnp, preds, mlp_output)
    plt.figure(figsize=(5, 5))
    sns.set(font_scale=1.2)
    sns.heatmap(mlpconfusion_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, square=True)
    # Configure the plot
    plt.xlabel("MLP")
    plt.ylabel("GPT")
    plt.title("Ouput Comparison")
    plt.tight_layout()
    plt.savefig(str(dataset_path / "mlp_gpt.png"))
    mlpul_file = dataset_path / "mlpul.txt"
    mlpur_file = dataset_path / "mlpur.txt"
    mlpll_file = dataset_path / "mlpll.txt"
    mlplr_file = dataset_path / "mlplr.txt"
    total = [mlpul_file, mlpur_file, mlpll_file, mlplr_file]
    res = [mlpul, mlpur, mlpll, mlplr]
    for r, f in zip(res, total):
        with f.open("w") as sentinel:
            pass
        for t in r:
            with f.open("a") as ff:
                ff.write(f"Sample {t + 1}\n")
                ff.write(prompts[t])
                ff.write("Ground truth: {}\n".format(label_names[gtnp[t]]))
                ff.write("GPT Prediction: {}\n".format(pred_texts[t]))
                ff.write("MLP Prediction: {}\n".format(label_names[mlp_output[t]]))
    
    
    gnnconfusion_matrix, gnnul, gnnur, gnnll, gnnlr = create_confusion_matrix(gtnp, preds, gnn_output)
    plt.figure(figsize=(5, 5))
    sns.set(font_scale=1.2)
    sns.heatmap(gnnconfusion_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, square=True)
    # Configure the plot
    plt.xlabel("gnn")
    plt.ylabel("GPT")
    plt.title("Ouput Comparison")
    plt.tight_layout()
    plt.savefig(str(dataset_path / "gnn_gpt.png"))
    gnnul_file = dataset_path / "gnnul.txt"
    gnnur_file = dataset_path / "gnnur.txt"
    gnnll_file = dataset_path / "gnnll.txt"
    gnnlr_file = dataset_path / "gnnlr.txt"
    total = [gnnul_file, gnnur_file, gnnll_file, gnnlr_file]
    res = [gnnul, gnnur, gnnll, gnnlr]
    for r, f in zip(res, total):
        with f.open("w") as sentinel:
            pass
        for t in r:
            with f.open("a") as ff:
                ff.write(f"Sample {t + 1}\n")
                ff.write(prompts[t])
                ff.write("Ground truth: {}\n".format(label_names[gtnp[t]]))
                ff.write("Prediction: {}\n".format(pred_texts[t]))
                ff.write("GNN Prediction: {}\n".format(label_names[gnn_output[t]]))


    return torch.tensor(preds)



def create_confusion_matrix(ground_truth_list, prediction_list_a, prediction_list_b):
    confusion_matrix = np.zeros((2, 2), dtype=int)
    ul, ur, ll, lr = [], [], [], []
    for i, (gt, pred_a, pred_b) in enumerate(zip(ground_truth_list, prediction_list_a, prediction_list_b)):
        if pred_a == gt and pred_b == gt:
            confusion_matrix[0, 0] += 1
            ul.append(i)
        elif pred_a == gt and pred_b != gt:
            confusion_matrix[0, 1] += 1
            ur.append(i)
        elif pred_a != gt and pred_b == gt:
            confusion_matrix[1, 0] += 1
            ll.append(i)
        elif pred_a != gt and pred_b != gt:
            confusion_matrix[1, 1] += 1
            lr.append(i)

    return confusion_matrix, ul, ur, ll, lr


def sample_from_data(data, sample_num, ego_size):
    nx_comm = to_networkx(data, to_undirected=True, remove_self_loops=True) 
    all_idxs = torch.arange(data.x.shape[0])
    test_node_idxs = all_idxs[data.test_masks[0]]
    if sample_num == -1:
        sampled_test_node_idxs = test_node_idxs
    else:
        sampled_test_node_idxs = test_node_idxs[torch.randperm(test_node_idxs.shape[0])[:sample_num]]
    ego_graphs = []
    centers = []
    for center_node_idx in tqdm(sampled_test_node_idxs):
        center_node_idx_int = center_node_idx.item()
        # ego_graph = nx.ego_graph(nx_comm, center_node_idx_int, radius=2, center = True, undirected=True)
        neighbors = list(nx_comm.neighbors(center_node_idx_int))
        subgraph_list = [center_node_idx_int]
        if len(neighbors) > ego_size:
            sample_node_list = random.sample(list(neighbors), ego_size)
            subgraph_list.extend(sample_node_list)
            subgraph_list = list(set(subgraph_list))
        else:
            subgraph_list.extend(neighbors)
        subgraph_list = list(set(subgraph_list))
        selected_context = nx_comm.subgraph(subgraph_list)
        ego_graphs.append(selected_context)
        centers.append(center_node_idx_int)
    # torch.save([ego_graphs, centers], osp.join(path, filename))
    return ego_graphs, centers



def neighbors(edge_index, node_id):
    row, col = edge_index 
    match_idx = torch.where(row == node_id)[0]
    neigh_nodes = col[match_idx]
    return neigh_nodes.tolist()

def get_sampled_nodes(data_obj, sample_num = -1):
    train_mask = data_obj.train_mask
    test_mask = data_obj.test_mask
    all_idxs = torch.arange(data_obj.y.shape[0])
    test_node_idxs = all_idxs[test_mask]
    train_node_idxs = all_idxs[train_mask]
    if sample_num == -1:
        sampled_test_node_idxs = test_node_idxs
    else:
        sampled_test_node_idxs = test_node_idxs[torch.randperm(test_node_idxs.shape[0])[:sample_num]]
    return sampled_test_node_idxs, train_node_idxs


def get_one_hop_neighbors(data_obj, sampled_test_node_idxs, sample_num = -1):
    ## if sample_nodes == -1, all test nodes within test masks will be considered
    neighbor_dict = {}
    for center_node_idx in sampled_test_node_idxs:
        center_node_idx = center_node_idx.item()
        neighbor_dict[center_node_idx] = neighbors(data_obj.edge_index, center_node_idx)
    return neighbor_dict

def get_two_hop_neighbors_no_multiplication(data_obj, sampled_test_node_idxs, sample_num = -1):
    neighbor_dict = {}
    # for center_node_idx in sampled_test_node_idxs:
    one_hop_neighbor_dict = get_one_hop_neighbors(data_obj, sampled_test_node_idxs)
    for key, value in one_hop_neighbor_dict.items():
        this_key_neigh = []
        second_hop_neighbor_dict = get_one_hop_neighbors(data_obj, torch.IntTensor(value))
        second_hop_neighbors = set(itertools.chain.from_iterable(second_hop_neighbor_dict.values()))
        second_hop_neighbors.discard(key)
        neighbor_dict[key] = sorted(list(second_hop_neighbors))
    return neighbor_dict


def get_two_hop_neighbors(data_obj, sampled_test_node_idxs, sample_nodes = -1):
    ## if sample_nodes == -1, all test nodes within test masks will be considered
    neighbor_dict = {}
    N = data_obj.x.shape[0]
    edge_index = data_obj.edge_index
    adj = to_torch_csr_tensor(edge_index, size=(N, N))
    edge_index2, _ = to_edge_index(adj @ adj)
    edge_index2, _ = remove_self_loops(edge_index2)
    edge_index = torch.cat([edge_index, edge_index2], dim=1)
    two_hop_edge_index, _ = coalesce(edge_index, None, N)
    for center_node_idx in sampled_test_node_idxs:
        center_node_idx = center_node_idx.item()
        neighbor_dict[center_node_idx] = neighbors(two_hop_edge_index, center_node_idx)
    return neighbor_dict


def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo":
        print("Warning: gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0301.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301")
    elif model == "gpt-4":
        print("Warning: gpt-4 may change over time. Returning num tokens assuming gpt-4-0314.")
        return num_tokens_from_messages(messages, model="gpt-4-0314")
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model == "gpt-4-0314":
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")
    num_tokens = len(encoding.encode(messages))
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


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

    products_mapping = {'Home & Kitchen': 'Home & Kitchen',
        'Health & Personal Care': 'Health & Personal Care',
        'Beauty': 'Beauty',
        'Sports & Outdoors': 'Sports & Outdoors',
        'Books': 'Books',
        'Patio, Lawn & Garden': 'Patio, Lawn & Garden',
        'Toys & Games': 'Toys & Games',
        'CDs & Vinyl': 'CDs & Vinyl',
        'Cell Phones & Accessories': 'Cell Phones & Accessories',
        'Grocery & Gourmet Food': 'Grocery & Gourmet Food',
        'Arts, Crafts & Sewing': 'Arts, Crafts & Sewing',
        'Clothing, Shoes & Jewelry': 'Clothing, Shoes & Jewelry',
        'Electronics': 'Electronics',
        'Movies & TV': 'Movies & TV',
        'Software': 'Software',
        'Video Games': 'Video Games',
        'Automotive': 'Automotive',
        'Pet Supplies': 'Pet Supplies',
        'Office Products': 'Office Products',
        'Industrial & Scientific': 'Industrial & Scientific',
        'Musical Instruments': 'Musical Instruments',
        'Tools & Home Improvement': 'Tools & Home Improvement',
        'Magazine Subscriptions': 'Magazine Subscriptions',
        'Baby Products': 'Baby Products',
        'label 25': 'label 25',
        'Appliances': 'Appliances',
        'Kitchen & Dining': 'Kitchen & Dining',
        'Collectibles & Fine Art': 'Collectibles & Fine Art',
        'All Beauty': 'All Beauty',
        'Luxury Beauty': 'Luxury Beauty',
        'Amazon Fashion': 'Amazon Fashion',
        'Computers': 'Computers',
        'All Electronics': 'All Electronics',
        'Purchase Circles': 'Purchase Circles',
        'MP3 Players & Accessories': 'MP3 Players & Accessories',
        'Gift Cards': 'Gift Cards',
        'Office & School Supplies': 'Office & School Supplies',
        'Home Improvement': 'Home Improvement',
        'Camera & Photo': 'Camera & Photo',
        'GPS & Navigation': 'GPS & Navigation',
        'Digital Music': 'Digital Music',
        'Car Electronics': 'Car Electronics',
        'Baby': 'Baby',
        'Kindle Store': 'Kindle Store',
        'Buy a Kindle': 'Buy a Kindle',
        'Furniture & D&#233;cor': 'Furniture & Decor',
        '#508510': '#508510'}
    if dataset is None:
        return arxiv_mapping, citeseer_mapping, pubmed_mapping, cora_mapping, products_mapping
    elif dataset.lower() == 'cora':
        return cora_mapping
    elif dataset.lower() == 'citeseer':
        return citeseer_mapping
    elif dataset.lower() == 'pubmed':
        return pubmed_mapping
    elif dataset.lower() == 'arxiv':
        return arxiv_mapping
    
def load_bow_config(filename):
    cora_config = {
        'max_tokens': 512,
        'max_words': 300,
        'num_classes': 7,
        'category_names': ["Rule Learning", "Neural Networks", "Case Based", "Genetic Algorithms", "Theory", "Reinforcement Learning", "Probabilistic Methods"]
    }

    citeseer_config = {
        'max_tokens': 512,
        'max_words': 300,
        'num_classes': 6,
        'category_names': ["Agents", "Machine Learning", "Information Retrieval", "Database", "Human Computer Interaction", "Artificial Intelligence"]
    }

    pubmed_config = {
        'max_tokens': 550,
        'max_words': 400,
        'num_classes': 3,
        'category_names': ['Diabetes Mellitus Experimental', 'Diabetes Mellitus Type 1', 'Diabetes Mellitus Type 2']
    }

    arxiv_config = {
        'max_tokens': 512,
        'max_words': 350,
        'num_classes': 40,
        'category_names': ['Artificial Intelligence', 'Computation and Language', 'Computational Complexity', 'Computational Engineering, Finance, and Science', 'Computational Geometry', 'Computer Science and Game Theory', 'Computer Vision and Pattern Recognition', 'Computers and Society', 'Cryptography and Security', 'Data Structures and Algorithms', 'Databases', 'Digital Libraries', 'Discrete Mathematics', 'Distributed, Parallel, and Cluster Computing', 'Emerging Technologies', 'Formal Languages and Automata Theory', 'General Literature', 'Graphics', 'Hardware Architecture', 'Human-Computer Interaction', 'Information Retrieval', 'Information Theory', 'Logic in Computer Science', 'Machine Learning', 'Mathematical Software', 'Multiagent Systems', 'Multimedia', 'Networking and Internet Architecture', 'Neural and Evolutionary Computing', 'Numerical Analysis', 'Operating Systems', 'Other Computer Science', 'Performance', 'Programming Languages', 'Robotics', 'Social and Information Networks', 'Software Engineering', 'Sound', 'Symbolic Computation', 'Systems and Control']
    }

    reddit_config = {
        'max_tokens': 550,
        'max_words': 400,
        'num_classes': 0,
        'category_names': []
    }
    

    if 'cora' in filename:
        return cora_config['max_tokens'], cora_config['max_words'], cora_config['num_classes'], cora_config['category_names']
    elif 'citeseer' in filename:
        return citeseer_config['max_tokens'], citeseer_config['max_words'], citeseer_config['num_classes'], citeseer_config['category_names']
    elif 'pubmed' in filename:
        return pubmed_config['max_tokens'], pubmed_config['max_words'], pubmed_config['num_classes'], pubmed_config['category_names']
    elif 'arxiv' in filename:
        return arxiv_config['max_tokens'], arxiv_config['max_words'], arxiv_config['num_classes'], arxiv_config['category_names']
    elif 'reddit' in filename:
        return reddit_config['max_tokens'], reddit_config['max_words'], reddit_config['num_classes'], reddit_config['category_names']
    

def few_shot():
    few_shot_samples = {
        "top1": {
            "cora": [
                """
                Paper:
                Stochastic pro-positionalization of non-determinate background knowledge. : It is a well-known fact that propositional learning algorithms require "good" features to perform well in practice. So a major step in data engineering for inductive learning is the construction of good features by domain experts. These features often represent properties of structured objects, where a property typically is the occurrence of a certain substructure having certain properties. To partly automate the process of "feature engineering", we devised an algorithm that searches for features which are defined by such substructures. The algorithm stochastically conducts a top-down search for first-order clauses, where each clause represents a binary feature. It differs from existing algorithms in that its search is not class-blind, and that it is capable of considering clauses ("context") of almost arbitrary length (size). Preliminary experiments are favorable, and support the view that this approach is promising.            Task: 
                There are following categories: 
                ['Rule Learning', 'Neural Networks', 'Case Based', 'Genetic Algorithms', 'Theory', 'Reinforcement Learning', 'Probabilistic Methods']
                Which category does this paper belong to?
                Output the most 1 possible category of this paper as a python list, like ['XX']
                Result:
                ['Rule Learning']
                """,
                """
                Paper:
                Neural networks and statistical models. : 
                Task: 
                There are following categories: 
                ['Rule Learning', 'Neural Networks', 'Case Based', 'Genetic Algorithms', 'Theory', 'Reinforcement Learning', 'Probabilistic Methods']
                Which category does this paper belong to?
                Output the most 1 possible category of this paper as a python list, like ['XX']
                Result:
                ['Neural Networks']
                """,
                """
                Paper:
                Looking at Markov samplers through CUM-SUM path plots: a simple diagnostic idea. : In this paper, we propose to monitor a Markov chain sampler using the cusum path plot of a chosen 1-dimensional summary statistic. We argue that the cusum path plot can bring out, more effectively than the sequential plot, those aspects of a Markov sampler which tell the user how quickly or slowly the sampler is moving around in its sample space, in the direction of the summary statistic. The proposal is then illustrated in four examples which represent situations where the cusum path plot works well and not well. Moreover, a rigorous analysis is given for one of the examples. We conclude that the cusum path plot is an effective tool for convergence diagnostics of a Markov sampler and for comparing different Markov samplers.
                Task: 
                There are following categories: 
                ['Rule Learning', 'Neural Networks', 'Case Based', 'Genetic Algorithms', 'Theory', 'Reinforcement Learning', 'Probabilistic Methods']
                Which category does this paper belong to?
                Output the most 1 possible category of this paper as a python list, like ['XX']
                Result:
                ['Case Based']
                """,
                """
                Paper:
                User\'s Guide to the PGAPack Parallel Genetic Algorithm Library Version 0.2. : 
                Task: 
                There are following categories: 
                ['Rule Learning', 'Neural Networks', 'Case Based', 'Genetic Algorithms', 'Theory', 'Reinforcement Learning', 'Probabilistic Methods']
                Which category does this paper belong to?
                Output the most 1 possible category of this paper as a python list, like ['XX']
                Result:
                ['Genetic Algorithms']
                """,
                """
                Paper:
                Optimality and domination in repeated games with bounded players. : We examine questions of optimality and domination in repeated stage games where one or both players may draw their strategies only from (perhaps different) computationally bounded sets. We also consider optimality and domination when bounded convergence rates of the infinite payoff. We develop a notion of a "grace period" to handle the problem of vengeful strategies.
                Task: 
                There are following categories: 
                ['Rule Learning', 'Neural Networks', 'Case Based', 'Genetic Algorithms', 'Theory', 'Reinforcement Learning', 'Probabilistic Methods']
                Which category does this paper belong to?
                Output the most 1 possible category of this paper as a python list, like ['XX']
                Result:
                ['Theory']
                """,
                """
                Paper:
                Evolving sensors in environments of controlled complexity. : 1 . Sensors represent a crucial link between the evolutionary forces shaping a species' relationship with its environment, and the individual's cognitive abilities to behave and learn. We report on experiments using a new class of "latent energy environments" (LEE) models to define environments of carefully controlled complexity which allow us to state bounds for random and optimal behaviors that are independent of strategies for achieving the behaviors. Using LEE's analytic basis for defining environments, we then use neural networks (NNets) to model individuals and a steady - state genetic algorithm to model an evolutionary process shaping the NNets, in particular their sensors. Our experiments consider two types of "contact" and "ambient" sensors, and variants where the NNets are not allowed to learn, learn via error correction from internal prediction, and via reinforcement learning. We find that predictive learning, even when using a larger repertoire of the more sophisticated ambient sensors, provides no advantage over NNets unable to learn. However, reinforcement learning using a small number of crude contact sensors does provide a significant advantage. Our analysis of these results points to a tradeoff between the genetic "robustness" of sensors and their informativeness to a learning system.
                Task: 
                There are following categories: 
                ['Rule Learning', 'Neural Networks', 'Case Based', 'Genetic Algorithms', 'Theory', 'Reinforcement Learning', 'Probabilistic Methods']
                Which category does this paper belong to?
                Output the most 1 possible category of this paper as a python list, like ['XX']
                Result:
                ['Reinforcement Learning']
                """,
                """
                Paper:
                Discovering structure in continuous variables using Bayesian networks. : We study Bayesian networks for continuous variables using nonlinear conditional density estimators. We demonstrate that useful structures can be extracted from a data set in a self-organized way and we present sampling techniques for belief update based on
                Task: 
                There are following categories: 
                ['Rule Learning', 'Neural Networks', 'Case Based', 'Genetic Algorithms', 'Theory', 'Reinforcement Learning', 'Probabilistic Methods']
                Which category does this paper belong to?
                Output the most 1 possible category of this paper as a python list, like ['XX']
                Result:
                ['Probabilistic Methods']
                """
            ],
            "citeseer": [
                """
                Paper:
                dQUOB: Managing Large Data Flows Using Dynamic Embedded Queries The dQUOB system satisfies client need for specific information from high-volume data streams. The data streams we speak of are the flow of data existing during large-scale visualizations, video streaming to large numbers of distributed users, and high volume business transactions. We introduces the notion of conceptualizing a data stream as a set of relational database tables so that a scientist can request information with an SQL-like query. Transformation or computation that often needs to be performed on the data en-route can be conceptualized ascomputation performed onconsecutive views of the data, with computation associated with each view. The dQUOB system moves the query code into the data stream as a quoblet; as compiled code. The relational database data model has the significant advantage of presenting opportunities for efficient reoptimizations of queries and sets of queries. Using examples from global atmospheric modeling, we illustrate the usefulness of the dQUOB system. We carry the examples through the experiments to establish the viability of the approach for high performance computing with a baseline benchmark. We define a cost-metric of end-to-end latency that can be used to determine realistic cases where optimization should be applied. Finally, we show that end-to-end latency can be controlled through a probability assigned to a query that a query will evaluate to true.            Task: 
                There are following categories: 
                ['Agents', 'Machine Learning', 'Information Retrieval', 'Database', 'Human Computer Interaction', 'Artificial Intelligence']
                Which category does this paper belong to?
                Output the most 1 possible categories of this paper as a python list, with the format ['XX']
                Result:
                ['Database']
                """,
                """
                Paper:
                Statistical Pattern Recognition: A Review AbstractÐThe primary goal of pattern recognition is supervised or unsupervised classification. Among the various frameworks in which pattern recognition has been traditionally formulated, the statistical approach has been most intensively studied and used in practice. More recently, neural network techniques and methods imported from statistical learning theory have been receiving increasing attention. The design of a recognition system requires careful attention to the following issues: definition of pattern classes, sensing environment, pattern representation, feature extraction and selection, cluster analysis, classifier design and learning, selection of training and test samples, and performance evaluation. In spite of almost 50 years of research and development in this field, the general problem of recognizing complex patterns with arbitrary orientation, location, and scale remains unsolved. New and emerging applications, such as data mining, web searching, retrieval of multimedia data, face recognition, and cursive handwriting recognition, require robust and efficient pattern recognition techniques. The objective of this review paper is to summarize and compare some of the well-known methods used in various stages of a pattern recognition system and identify research topics and applications which are at the forefront of this exciting and challenging field.            
                There are following categories: 
                ['Agents', 'Machine Learning', 'Information Retrieval', 'Database', 'Human Computer Interaction', 'Artificial Intelligence']
                Which category does this paper belong to?
                Output the most 1 possible categories of this paper as a python list, with the format ['XX']
                Result:
                ['Machine Learning']
                """,
                """
                Paper:
                Group Task Analysis for Groupware Usability Evaluations Techniques for inspecting the usability of groupware applications have recently been proposed. These techniques focus on the mechanics of collaboration rather than the work context in which a system is used, and offer time and cost savings by not requiring actual users or fully-functional prototypes. Although these techniques are valuable, adding information about task and work context could improve the quality of inspection results. We introduce a method for analysing group tasks that can be used to add context to discount groupware evaluation techniques. Our method allows for the specification of collaborative scenarios and tasks by considering the mechanics of collaboration, levels of coupling during task performance, and variability in task execution. We describe how this type of task analysis could be used in a new inspection technique based on cognitive walkthrough.
                There are following categories: 
                ['Agents', 'Machine Learning', 'Information Retrieval', 'Database', 'Human Computer Interaction', 'Artificial Intelligence']
                Which category does this paper belong to?
                Output the most 1 possible categories of this paper as a python list, with the format ['XX']
                Result:
                ['Human Computer Interaction']            
                """,
                """
                Paper:
                Emergent Cooperative Goal-Satisfaction in Large Scale Automated-Agent Systems Cooperation among autonomous agents has been discussed in the DAI community for several years. Papers about cooperation [6, 45], negotiation [33], distributed planning [5], and coalition formation [28, 48], have provided a variety of approaches and several algorithms and solutions to situations wherein cooperation is possible. However, the case of cooperation in large-scale multi-agent systems (MAS) has not been thoroughly examined. Therefore, in this paper we present a framework for cooperative goal-satisfaction in large-scale environments focusing on a low complexity physics-oriented approach. The multi-agent systems with which we deal are modeled by a physics-oriented model. According to the model, MAS inherit physical properties, and therefore the evolution of the computational systems is similar to the evolution of physical systems. To enable implementation of the model, we provide a detailed algorithm to be used by a single agent within the system. The model and the algorithm are a...
                There are following categories: 
                ['Agents', 'Machine Learning', 'Information Retrieval', 'Database', 'Human Computer Interaction', 'Artificial Intelligence']
                Which category does this paper belong to?
                Output the most 1 possible categories of this paper as a python list, with the format ['XX']
                Result:
                ['Agents']    
                """,
                """
                Paper:
                Managing Robot Autonomy and Interactivity Using Motives and Visual Communication An autonomous mobile robot operating in everyday life conditions will have to face a huge variety of situations and to interact with other agents (living or artificial). Such a robot needs flexible and robust methods for managing its goals and for adapting its control mechanisms to face the contingencies of the world. It also needs to communicate with others in order to get useful information about the world. This paper describes an approach based on a general architecture and on internal variables called `motives' to manage the goals of an autonomous robot. These variables are also used as a basis for communication using a visual communication system. Experiments using a vision- and sonar-based Pioneer I robot, equipped with a visual signaling device, are presented.  1 Introduction  Designing an autonomous mobile robot to operate in unmodified environments, i.e., environments that have not been specifically engineered for the robot, is a very challenging problems. Dynamic and unpredic...
                There are following categories: 
                ['Agents', 'Machine Learning', 'Information Retrieval', 'Database', 'Human Computer Interaction', 'Artificial Intelligence']
                Which category does this paper belong to?
                Output the most 1 possible categories of this paper as a python list, with the format ['XX']
                Result:
                ['Artificial Intelligence']    
                """,
                """
                Paper:
                Stable Algorithms for Link Analysis The Kleinberg HITS and the Google PageRank algorithms are eigenvector methods for identifying "authoritative" or "influential" articles, given hyperlink or citation information. That such algorithms should give reliable or consistent answers is surely a desideratum, and in [10], we analyzed when they can be expected to give stable rankings under small perturbations to the linkage patterns. In this paper, we extend the analysis and show how it gives insight into ways of designing stable link analysis methods. This in turn motivates two new algorithms, whose performance we study empirically using citation data and web hyperlink data. 1.
                There are following categories: 
                ['Agents', 'Machine Learning', 'Information Retrieval', 'Database', 'Human Computer Interaction', 'Artificial Intelligence']
                Which category does this paper belong to?
                Output the most 1 possible categories of this paper as a python list, with the format ['XX']
                Result:
                ['Information Retrieval']    
                """
            ],
            "pubmed": [
                """
                Title: Glycemic response to newly initiated diabetes therapies.
                Abstract: OBJECTIVE: The glycemic response to antihyperglycemic therapies for type 2 diabetes has been thoroughly evaluated in randomized controlled trials, but inadequately studied in real-world settings. STUDY DESIGN: We studied glycemic response among 15 126 type 2 diabetic patients who initiated any single new antihyperglycemic agent (metformin, sulfonylureas, thiazolidinediones, or insulin added to medical nutrition therapy or to existing diabetes therapies) during 1999-2000 within Kaiser Permanente of Northern California, an integrated healthcare delivery system. METHODS: Pre-post (3-12 months after initiation) change in glycosylated hemoglobin (A1C) was analyzed using ANCOVA (analysis of covariance) models adjusted for baseline A1C, concurrent (ongoing) antihyperglycemic therapy, demographics, health behaviors, medication adherence, clinical factors, and processes of care. RESULTS: Mean A1C was 9.01% (95% confidence interval [CI] 8.98%-9.04%) before therapy initiation and 7.87% (95% CI 7.85%-7.90%) 3 to 12 months after initiation (mean A1C reduction 1.14 percentage points; 95% CI 1.11-1.17). Overall, 30.2% (95% CI 29.2%-31.1%) of patients achieved glycemic target (A1C < 7%). Although baseline disease severity and concurrent therapies differed greatly across therapeutic classes, after adjustment for these baseline clinical characteristics, no significant differences were noted in glucose-lowering effect across therapeutic classes. Treatment effects did not differ by age, race, diabetes duration, obesity, or level of renal function. CONCLUSIONS: Metformin, sulfonylures, thiazolidinediones, and insulin were equally effective in improving glucose control. Nonetheless, most patients failed to achieve the glycemic target. Findings suggest that, to keep up with progressive worsening of glycemic control, patients and providers must commit to earlier, more aggressive therapy intensification, triggered promptly after A1C exceeds the recommended glycemic target.
                Task: 
                There are following categories: 
                ['Diabetes Mellitus, Experimental', 'Diabetes Mellitus Type 1', 'Diabetes Mellitus Type 2']
                Which category does this paper belong to?
                Output the most 1 possible category of this paper as a python list, like ['XX']
                Result:
                ['Diabetes Mellitus Type 2']
                """, 
                """
                Title: Thyroid autoimmunity in children and adolescents with type 1 diabetes: a multicenter survey.
                Abstract: OBJECTIVE: To investigate thyroid autoimmunity in a very large nationwide cohort of children and adolescents with type 1 diabetes. RESEARCH DESIGN AND METHODS: Data were analyzed from 17,749 patients with type 1 diabetes aged 0.1-20 years who were treated in 118 pediatric diabetes centers in Germany and Austria. Antibodies to thyroglobulin (anti-TG) and thyroperoxidase (anti-TPO) were measured and documented at least once in 7,097 patients. A total of 49.5% of these patients were boys, the mean age was 12.4 years (range 0.3-20.0 years), and the mean duration of diabetes was 4.5 years (range 0.0-19.5 years). A titer exceeding 100 units/ml or 1:100 was considered significantly elevated. RESULTS: In 1,530 patients, thyroid antibody levels were elevated on at least one occasion, whereas 5,567 were antibody-negative during the observation period. Patients with thyroid antibodies were significantly older (P < 0.001), had a longer duration of diabetes (P < 0.001), and developed diabetes later in life (P < 0.001) than those without antibodies. A total of 63% of patients with positive antibodies were girls, compared with 45% of patients without antibodies (P < 0.001). The prevalence of significant thyroid antibody titers increased with increasing age; the highest prevalence was in the 15- to 20-year age group (anti-TPO: 16.9%, P < 0.001; anti-TG: 12.8%, P < 0.001). Thyroid-stimulating hormone (TSH) levels were higher in patients with thyroid autoimmunity (3.34 microU/ml, range 0.0-615.0 microU/ml) than in control subjects (1.84 microU/ml, range 0.0-149.0 microU/ml) (P < 0.001). Even higher TSH levels were observed in patients with both anti-TPO and anti-TG (4.55 microU/ml, range 0.0-197.0 microU/ml). CONCLUSIONS: Thyroid autoimmunity seems to be particularly common in girls with diabetes during the second decade of life and may be associated with elevated TSH levels, indicating subclinical hypothyroidism.
                Task: 
                There are following categories: 
                ['Diabetes Mellitus, Experimental', 'Diabetes Mellitus Type 1', 'Diabetes Mellitus Type 2']
                Which category does this paper belong to?
                Output the most 1 possible category of this paper as a python list, like ['XX']
                Result:
                ['Diabetes Mellitus Type 1']
                """,
                """
                Title: Specific changes of somatostatin mRNA expression in the frontal cortex and hippocampus of diabetic rats.
                Abstract: Abstract Most current studies of diabetic encephalopathy have focused on brain blood flow and metabolism, but there has been little research on the influence of diabetes on brain tissue and the causes of chronic diabetic encephalopathy. The technique of molecular biology makes it possible to explore the mechanism of chronic diabetic encephalopathy by testing the distribution of somatostatin in the brain. We have therefore analysed, by in situ hybridization histochemistry, the changes in somatostatin (SST) mRNA in the frontal cortex and hippocampus of rats made diabetic by the injection of streptozotocin. Ten Sprague-Dawley control rats were compared with ten streptozotocin-induced diabetic rats. The weight, blood glucose and urine glucose did not differ between the two groups before the injection of streptozotocin. Four weeks after the injection of streptozotocin the weight, blood glucose and urine glucose of the diabetic rats were, respectively, 199.1 +/- 15.6 g, 23.7 +/- 3.25 mmol L(-1) and (++) to (+++) whereas those of the control group were 265.5 +/- 30.3 g, 4.84 +/- 0.63 mmol L(-1) and (-). Somatostatin mRNA was reduced in the diabetic rats. The number of SST mRNA-positive neurons and the optical density of positive cells in the hippocampus and frontal cortex of the diabetic rats were 80.6 +/- 17.5 mm(-2) and 76.5 +/- 17.6 compared with 150.5 +/- 21.1 mm(-2) and 115.1 +/- 18.5 in the control rats. The induction of diabetes is thus associated with a decreased expression of SST mRNA in the hippocampus and frontal cortex, which might be an important component of chronic diabetic encephalopathy
                Task: 
                There are following categories: 
                ['Diabetes Mellitus, Experimental', 'Diabetes Mellitus Type 1', 'Diabetes Mellitus Type 2']
                Which category does this paper belong to?
                Output the most 1 possible category of this paper as a python list, like ['XX']
                Result:
                ['Diabetes Mellitus, Experimental']
                """
            ],
            "arxiv": [
                """
                Paper:
                the exp log normal form of types decomposing extensional equality and representing terms compactly Lambda calculi with algebraic data types lie at the core of functional programming languages and proof assistants, but conceal at least two fundamental theoretical problems already in the presence of the simplest non-trivial data type, the sum type. First, we do not know of an explicit and implemented algorithm for deciding the beta-eta-equality of terms---and this in spite of the first decidability results proven two decades ago. Second, it is not clear how to decide when two types are essentially the same, i.e. isomorphic, in spite of the meta-theoretic results on decidability of the isomorphism.     In this paper, we present the exp-log normal form of types---derived from the representation of exponential polynomials via the unary exponential and logarithmic functions---that any type built from arrows, products, and sums, can be isomorphically mapped to. The type normal form can be used as a simple heuristic for deciding type isomorphism, thanks to the fact that it is a systematic application of the high-school identities.   We then show that the type normal form allows to reduce the standard beta-eta equational theory of the lambda calculus to a specialized version of itself, while preserving completeness of the equality on terms.     We end by describing an alternative representation of normal terms of the lambda calculus with sums, together with a Coq-implemented converter into/from our new term calculus. The difference with the only other previously implemented heuristic for deciding interesting instances of eta-equality by Balat, Di Cosmo, and Fiore, is that we exploits the type information of terms substantially and this often allows us to obtain a canonical representation of terms without performing a sophisticated term analyses.
                Task: 
                There are following categories: 
                ['Numerical Analysis', 'Multimedia', 'Logic in Computer Science', 'Computers and Society', 'Cryptography and Security', 'Distributed, Parallel, and Cluster Computing', 'Human-Computer Interaction', 'Computational Engineering, Finance, and Science', 'Networking and Internet Architecture', 'Computational Complexity', 'Artificial Intelligence', 'Multiagent Systems', 'General Literature', 'Neural and Evolutionary Computing', 'Symbolic Computation', 'Hardware Architecture', 'Computer Vision and Pattern Recognition', 'Graphics', 'Emerging Technologies', 'Systems and Control', 'Computational Geometry', 'Other Computer Science', 'Programming Languages', 'Software Engineering', 'Machine Learning', 'Sound', 'Social and Information Networks', 'Robotics', 'Information Theory', 'Performance', 'Computation and Language', 'Information Retrieval', 'Mathematical Software', 'Formal Languages and Automata Theory', 'Data Structures and Algorithms', 'Operating Systems', 'Computer Science and Game Theory', 'Databases', 'Digital Libraries', 'Discrete Mathematics']
                Which category does this paper belong to?
                Output the most 1 possible category of this paper as a python list, like ['XX']
                Result:
                ['Logic in Computer Science']
                """,
                """
                Paper:
                the ubuntu dialogue corpus a large dataset for research in unstructured multi turn dialogue systems This paper introduces the Ubuntu Dialogue Corpus, a dataset containing almost 1 million multi-turn dialogues, with a total of over 7 million utterances and 100 million words. This provides a unique resource for research into building dialogue managers based on neural language models that can make use of large amounts of unlabeled data. The dataset has both the multi-turn property of conversations in the Dialog State Tracking Challenge datasets, and the unstructured nature of interactions from microblog services such as Twitter. We also describe two neural learning architectures suitable for analyzing this dataset, and provide benchmark performance on the task of selecting the best next response.
                Task:
                Which arxiv cs subcategories does this paper belong to?
                Output the most 1 possible categories of this paper as a python list, with the format ['XX']
                Result:
                ['Computation and Language']
                """,
                """
                Paper:
                non memoryless analog network coding in two way relay channel Physical-layer Network Coding (PNC) can significantly improve the throughput of two-way relay channels. An interesting variant of PNC is Analog Network Coding (ANC). Almost all ANC schemes proposed to date, however, operate in a symbol by symbol manner (memoryless) and cannot exploit the redundant information in channel-coded packets to enhance performance. This paper proposes a non-memoryless ANC scheme. In particular, we design a soft-input soft-output decoder for the relay node to process the superimposed packets from the two end nodes to yield an estimated MMSE packet for forwarding back to the end nodes. Our decoder takes into account the correlation among different symbols in the packets due to channel coding, and provides significantly improved MSE performance. Our analysis shows that the SNR improvement at the relay node is lower bounded by 1/R (R is the code rate) with the simplest LDPC code (repeat code). The SNR improvement is also verified by numerical simulation with LDPC code. Our results indicate that LDPC codes of different degrees are preferred in different SNR regions. Generally speaking, smaller degrees are preferred for lower SNRs.            
                Task: 
                There are following categories: 
                ['Numerical Analysis', 'Multimedia', 'Logic in Computer Science', 'Computers and Society', 'Cryptography and Security', 'Distributed, Parallel, and Cluster Computing', 'Human-Computer Interaction', 'Computational Engineering, Finance, and Science', 'Networking and Internet Architecture', 'Computational Complexity', 'Artificial Intelligence', 'Multiagent Systems', 'General Literature', 'Neural and Evolutionary Computing', 'Symbolic Computation', 'Hardware Architecture', 'Computer Vision and Pattern Recognition', 'Graphics', 'Emerging Technologies', 'Systems and Control', 'Computational Geometry', 'Other Computer Science', 'Programming Languages', 'Software Engineering', 'Machine Learning', 'Sound', 'Social and Information Networks', 'Robotics', 'Information Theory', 'Performance', 'Computation and Language', 'Information Retrieval', 'Mathematical Software', 'Formal Languages and Automata Theory', 'Data Structures and Algorithms', 'Operating Systems', 'Computer Science and Game Theory', 'Databases', 'Digital Libraries', 'Discrete Mathematics']
                Which category does this paper belong to?
                Output the most 1 possible category of this paper as a python list, like ['XX']
                Result:
                ['Information Theory']
                """,
                """
                Paper:
                approximate correlation clustering using same cluster queries Ashtiani et al. (NIPS 2016) introduced a semi-supervised framework for clustering (SSAC) where a learner is allowed to make same-cluster queries. More specifically, in their model, there is a query oracle that answers queries of the form given any two vertices, do they belong to the same optimal cluster?. Ashtiani et al. showed the usefulness of such a query framework by giving a polynomial time algorithm for the k-means clustering problem where the input dataset satisfies some separation condition. Ailon et al. extended the above work to the approximation setting by giving an efficient (1+\eps)-approximation algorithm for k-means for any small \eps > 0 and any dataset within the SSAC framework. In this work, we extend this line of study to the correlation clustering problem. Correlation clustering is a graph clustering problem where pairwise similarity (or dissimilarity) information is given for every pair of vertices and the objective is to partition the vertices into clusters that minimise the disagreement (or maximises agreement) with the pairwise information given as input. These problems are popularly known as MinDisAgree and MaxAgree problems, and MinDisAgree[k] and MaxAgree[k] are versions of these problems where the number of optimal clusters is at most k. There exist Polynomial Time Approximation Schemes (PTAS) for MinDisAgree[k] and MaxAgree[k] where the approximation guarantee is (1+\eps) for any small \eps and the running time is polynomial in the input parameters but exponential in k and 1/\eps. We obtain an (1+\eps)-approximation algorithm for any small \eps with running time that is polynomial in the input parameters and also in k and 1/\eps. We also give non-trivial upper and lower bounds on the number of same-cluster queries, the lower bound being based on the Exponential Time Hypothesis (ETH).
                Task: 
                There are following categories: 
                ['Numerical Analysis', 'Multimedia', 'Logic in Computer Science', 'Computers and Society', 'Cryptography and Security', 'Distributed, Parallel, and Cluster Computing', 'Human-Computer Interaction', 'Computational Engineering, Finance, and Science', 'Networking and Internet Architecture', 'Computational Complexity', 'Artificial Intelligence', 'Multiagent Systems', 'General Literature', 'Neural and Evolutionary Computing', 'Symbolic Computation', 'Hardware Architecture', 'Computer Vision and Pattern Recognition', 'Graphics', 'Emerging Technologies', 'Systems and Control', 'Computational Geometry', 'Other Computer Science', 'Programming Languages', 'Software Engineering', 'Machine Learning', 'Sound', 'Social and Information Networks', 'Robotics', 'Information Theory', 'Performance', 'Computation and Language', 'Information Retrieval', 'Mathematical Software', 'Formal Languages and Automata Theory', 'Data Structures and Algorithms', 'Operating Systems', 'Computer Science and Game Theory', 'Databases', 'Digital Libraries', 'Discrete Mathematics']
                Which category does this paper belong to?
                Output the most 1 possible category of this paper as a python list, like ['XX']
                Result:
                ['Data Structures and Algorithms']
                """,
                """
                Paper:
                2 stack pushall sortable permutations In the 60's, Knuth introduced stack-sorting and serial compositions of stacks. In particular, one significant question arise out of the work of Knuth: how to decide efficiently if a given permutation is sortable with 2 stacks in series? Whether this problem is polynomial or NP-complete is still unanswered yet. In this article we introduce 2-stack pushall permutations which form a subclass of 2-stack sortable permutations and show that these two classes are closely related. Moreover, we give an optimal O(n^2) algorithm to decide if a given permutation of size n is 2-stack pushall sortable and describe all its sortings. This result is a step to the solve the general $2$-stack sorting problem in polynomial time.            
                Task: 
                There are following categories: 
                ['Numerical Analysis', 'Multimedia', 'Logic in Computer Science', 'Computers and Society', 'Cryptography and Security', 'Distributed, Parallel, and Cluster Computing', 'Human-Computer Interaction', 'Computational Engineering, Finance, and Science', 'Networking and Internet Architecture', 'Computational Complexity', 'Artificial Intelligence', 'Multiagent Systems', 'General Literature', 'Neural and Evolutionary Computing', 'Symbolic Computation', 'Hardware Architecture', 'Computer Vision and Pattern Recognition', 'Graphics', 'Emerging Technologies', 'Systems and Control', 'Computational Geometry', 'Other Computer Science', 'Programming Languages', 'Software Engineering', 'Machine Learning', 'Sound', 'Social and Information Networks', 'Robotics', 'Information Theory', 'Performance', 'Computation and Language', 'Information Retrieval', 'Mathematical Software', 'Formal Languages and Automata Theory', 'Data Structures and Algorithms', 'Operating Systems', 'Computer Science and Game Theory', 'Databases', 'Digital Libraries', 'Discrete Mathematics']
                Which category does this paper belong to?
                Output the most 1 possible category of this paper as a python list, like ['XX']
                Result:
                ['Discrete Mathematics']
                """,
                """
                Paper:
                that s sick dude automatic identification of word sense change across different timescales In this paper, we propose an unsupervised method to identify noun sense changes based on rigorous analysis of time-varying text data available in the form of millions of digitized books. We construct distributional thesauri based networks from data at different time points and cluster each of them separately to obtain word-centric sense clusters corresponding to the different time points. Subsequently, we compare these sense clusters of two different time points to find if (i) there is birth of a new sense or (ii) if an older sense has got split into more than one sense or (iii) if a newer sense has been formed from the joining of older senses or (iv) if a particular sense has died. We conduct a thorough evaluation of the proposed methodology both manually as well as through comparison with WordNet. Manual evaluation indicates that the algorithm could correctly identify 60.4% birth cases from a set of 48 randomly picked samples and 57% split/join cases from a set of 21 randomly picked samples. Remarkably, in 44% cases the birth of a novel sense is attested by WordNet, while in 46% cases and 43% cases split and join are respectively confirmed by WordNet. Our approach can be applied for lexicography, as well as for applications like word sense disambiguation or semantic search.            
                Task: 
                There are following categories: 
                ['Numerical Analysis', 'Multimedia', 'Logic in Computer Science', 'Computers and Society', 'Cryptography and Security', 'Distributed, Parallel, and Cluster Computing', 'Human-Computer Interaction', 'Computational Engineering, Finance, and Science', 'Networking and Internet Architecture', 'Computational Complexity', 'Artificial Intelligence', 'Multiagent Systems', 'General Literature', 'Neural and Evolutionary Computing', 'Symbolic Computation', 'Hardware Architecture', 'Computer Vision and Pattern Recognition', 'Graphics', 'Emerging Technologies', 'Systems and Control', 'Computational Geometry', 'Other Computer Science', 'Programming Languages', 'Software Engineering', 'Machine Learning', 'Sound', 'Social and Information Networks', 'Robotics', 'Information Theory', 'Performance', 'Computation and Language', 'Information Retrieval', 'Mathematical Software', 'Formal Languages and Automata Theory', 'Data Structures and Algorithms', 'Operating Systems', 'Computer Science and Game Theory', 'Databases', 'Digital Libraries', 'Discrete Mathematics']
                Which category does this paper belong to?
                Output the most 1 possible category of this paper as a python list, like ['XX']
                Result:
                ['Computation and Language']
                """
            ],
            "products": [
                """
                Product Description:
                Little Book of Whittling, The: Passing Time on the trail, on the Porch, and Under the Stars (Woodcarving Illustrated Books) "[A] plot to get us all into the outdoors...enjoying ourselves while building skills! --"Tactical Knives Magazine"            Task: 
                There are following categories: 
                ['Home & Kitchen', 'Health & Personal Care', 'Beauty', 'Sports & Outdoors', 'Books', 'Patio, Lawn & Garden', 'Toys & Games', 'CDs & Vinyl', 'Cell Phones & Accessories', 'Grocery & Gourmet Food', 'Arts, Crafts & Sewing', 'Clothing, Shoes & Jewelry', 'Electronics', 'Movies & TV', 'Software', 'Video Games', 'Automotive', 'Pet Supplies', 'Office Products', 'Industrial & Scientific', 'Musical Instruments', 'Tools & Home Improvement', 'Magazine Subscriptions', 'Baby Products', 'label 25', 'Appliances', 'Kitchen & Dining', 'Collectibles & Fine Art', 'All Beauty', 'Luxury Beauty', 'Amazon Fashion', 'Computers', 'All Electronics', 'Purchase Circles', 'MP3 Players & Accessories', 'Gift Cards', 'Office & School Supplies', 'Home Improvement', 'Camera & Photo', 'GPS & Navigation', 'Digital Music', 'Car Electronics', 'Baby', 'Kindle Store', 'Buy a Kindle', 'Furniture & Decor', '#508510']
                Which category does this product from Amazon belong to?
                Output the most 1 possible category of this product as a python list, like ['XX']
                Result:
                ['Books']
                """,
                """
                Product Description:
                Zojirushi BB-CEC20 Home Bakery Supreme 2-Pound-Loaf Breadmaker, Black 
                Task: 
                There are following categories: 
                ['Home & Kitchen', 'Health & Personal Care', 'Beauty', 'Sports & Outdoors', 'Books', 'Patio, Lawn & Garden', 'Toys & Games', 'CDs & Vinyl', 'Cell Phones & Accessories', 'Grocery & Gourmet Food', 'Arts, Crafts & Sewing', 'Clothing, Shoes & Jewelry', 'Electronics', 'Movies & TV', 'Software', 'Video Games', 'Automotive', 'Pet Supplies', 'Office Products', 'Industrial & Scientific', 'Musical Instruments', 'Tools & Home Improvement', 'Magazine Subscriptions', 'Baby Products', 'label 25', 'Appliances', 'Kitchen & Dining', 'Collectibles & Fine Art', 'All Beauty', 'Luxury Beauty', 'Amazon Fashion', 'Computers', 'All Electronics', 'Purchase Circles', 'MP3 Players & Accessories', 'Gift Cards', 'Office & School Supplies', 'Home Improvement', 'Camera & Photo', 'GPS & Navigation', 'Digital Music', 'Car Electronics', 'Baby', 'Kindle Store', 'Buy a Kindle', 'Furniture & Decor', '#508510']
                Which category does this product from Amazon belong to?
                Output the most 1 possible category of this product as a python list, like ['XX']
                Result:
                ['Home & Kitchen']
                """,
                """
                Product Description:
                "Sentry HC Groom'n Comb/with Catnip Pouch" Virbac Kitty Korner Komber Self-Grooming Aid for Cats is made of durable, high-impact grade plastic. Attaches easily to wall or corner. Simple to remove and easy to clean. Appeals to cat's natural rubbing instinct.
                Task: 
                There are following categories: 
                ['Home & Kitchen', 'Health & Personal Care', 'Beauty', 'Sports & Outdoors', 'Books', 'Patio, Lawn & Garden', 'Toys & Games', 'CDs & Vinyl', 'Cell Phones & Accessories', 'Grocery & Gourmet Food', 'Arts, Crafts & Sewing', 'Clothing, Shoes & Jewelry', 'Electronics', 'Movies & TV', 'Software', 'Video Games', 'Automotive', 'Pet Supplies', 'Office Products', 'Industrial & Scientific', 'Musical Instruments', 'Tools & Home Improvement', 'Magazine Subscriptions', 'Baby Products', 'label 25', 'Appliances', 'Kitchen & Dining', 'Collectibles & Fine Art', 'All Beauty', 'Luxury Beauty', 'Amazon Fashion', 'Computers', 'All Electronics', 'Purchase Circles', 'MP3 Players & Accessories', 'Gift Cards', 'Office & School Supplies', 'Home Improvement', 'Camera & Photo', 'GPS & Navigation', 'Digital Music', 'Car Electronics', 'Baby', 'Kindle Store', 'Buy a Kindle', 'Furniture & Decor', '#508510']
                Which category does this product from Amazon belong to?
                Output the most 1 possible category of this product as a python list, like ['XX']
                Result:
                ['Pet Supplies']
                """,
                """
                Product Description:
                "Columbia Women's Silver Ridge Plaid Long Sleeve Shirt" A Columbia classic, this breezy plaid shirt offers built-in UPF 40 sun block to provide protection from the elements while keeping you dry and comfortable during active days in the outdoors.
                Task: 
                There are following categories: 
                ['Home & Kitchen', 'Health & Personal Care', 'Beauty', 'Sports & Outdoors', 'Books', 'Patio, Lawn & Garden', 'Toys & Games', 'CDs & Vinyl', 'Cell Phones & Accessories', 'Grocery & Gourmet Food', 'Arts, Crafts & Sewing', 'Clothing, Shoes & Jewelry', 'Electronics', 'Movies & TV', 'Software', 'Video Games', 'Automotive', 'Pet Supplies', 'Office Products', 'Industrial & Scientific', 'Musical Instruments', 'Tools & Home Improvement', 'Magazine Subscriptions', 'Baby Products', 'label 25', 'Appliances', 'Kitchen & Dining', 'Collectibles & Fine Art', 'All Beauty', 'Luxury Beauty', 'Amazon Fashion', 'Computers', 'All Electronics', 'Purchase Circles', 'MP3 Players & Accessories', 'Gift Cards', 'Office & School Supplies', 'Home Improvement', 'Camera & Photo', 'GPS & Navigation', 'Digital Music', 'Car Electronics', 'Baby', 'Kindle Store', 'Buy a Kindle', 'Furniture & Decor', '#508510']
                Which category does this product from Amazon belong to?
                Output the most 1 possible category of this product as a python list, like ['XX']
                Result:
                ['Sports & Outdoors']
                """,
                """
                Product Description:
                Papo 53012 Grey Wolf Figure The Papo toy line features beatifully crafted figurines and animals. Papo toys come in a wide variety of colors, all hand painted and bursting with imagination. We carry a wide selection for hours of play. Scale 1:20 True to life modeling. Meticulously hand painted figurines.
                Task: 
                There are following categories: 
                ['Home & Kitchen', 'Health & Personal Care', 'Beauty', 'Sports & Outdoors', 'Books', 'Patio, Lawn & Garden', 'Toys & Games', 'CDs & Vinyl', 'Cell Phones & Accessories', 'Grocery & Gourmet Food', 'Arts, Crafts & Sewing', 'Clothing, Shoes & Jewelry', 'Electronics', 'Movies & TV', 'Software', 'Video Games', 'Automotive', 'Pet Supplies', 'Office Products', 'Industrial & Scientific', 'Musical Instruments', 'Tools & Home Improvement', 'Magazine Subscriptions', 'Baby Products', 'label 25', 'Appliances', 'Kitchen & Dining', 'Collectibles & Fine Art', 'All Beauty', 'Luxury Beauty', 'Amazon Fashion', 'Computers', 'All Electronics', 'Purchase Circles', 'MP3 Players & Accessories', 'Gift Cards', 'Office & School Supplies', 'Home Improvement', 'Camera & Photo', 'GPS & Navigation', 'Digital Music', 'Car Electronics', 'Baby', 'Kindle Store', 'Buy a Kindle', 'Furniture & Decor', '#508510']
                Which category does this product from Amazon belong to?
                Output the most 1 possible category of this product as a python list, like ['XX']
                Result:
                ['Toys & Games']
                """,
                """
                Product Description:
                CS Hyde Metalized Mylar Tape with Acrylic Adhesive, 2.2mm Thick, Gold Color, 0.5&quot; x 72 yds CS Hyde Metalized Mylar Tape maintains excellent solvent resistance and long-term aging properties.&#xA0; This product is dimensionally stable and flexible to 302F.&#xA0; It&#xA0; features high reflectivity of both light and heat radiation, excellent heat resistance and easy unwind characteristics. 75% elongation, which measures how much you can stretch the tape before snapping. 45 oz/inch adhesive strength, which measures how well the item sticks to the substrate.
                Task: 
                There are following categories: 
                ['Home & Kitchen', 'Health & Personal Care', 'Beauty', 'Sports & Outdoors', 'Books', 'Patio, Lawn & Garden', 'Toys & Games', 'CDs & Vinyl', 'Cell Phones & Accessories', 'Grocery & Gourmet Food', 'Arts, Crafts & Sewing', 'Clothing, Shoes & Jewelry', 'Electronics', 'Movies & TV', 'Software', 'Video Games', 'Automotive', 'Pet Supplies', 'Office Products', 'Industrial & Scientific', 'Musical Instruments', 'Tools & Home Improvement', 'Magazine Subscriptions', 'Baby Products', 'label 25', 'Appliances', 'Kitchen & Dining', 'Collectibles & Fine Art', 'All Beauty', 'Luxury Beauty', 'Amazon Fashion', 'Computers', 'All Electronics', 'Purchase Circles', 'MP3 Players & Accessories', 'Gift Cards', 'Office & School Supplies', 'Home Improvement', 'Camera & Photo', 'GPS & Navigation', 'Digital Music', 'Car Electronics', 'Baby', 'Kindle Store', 'Buy a Kindle', 'Furniture & Decor', '#508510']
                Which category does this product from Amazon belong to?
                Output the most 1 possible category of this product as a python list, like ['XX']
                Result:
                ['Tools & Home Improvement']
                """
            ]
        }, 
        "cot": {
            "cora": [
                """
    This paper falls under the "Rule Learning" category because it presents an algorithm that stochastically conducts a top-down search for first-order clauses, where each clause (rule) represents a binary feature, essentially automating the process of "feature engineering", which is a characteristic methodology of rule-based machine learning approaches.
                """, 
                """
    This paper belongs to the "Neural Networks" category because the title explicitly mentions "Neural networks," indicating that the focus or significant part of the study involves the exploration, use, or discussion of neural network models.
                """,
                """
    This paper belongs to the "Case Based" category because it discusses memory-based techniques, which are a form of case-based reasoning in artificial intelligence, used in an interactive computer-aided design system (demex) to store, organize, retrieve, and reuse experiential knowledge for aiding designers in problem exploration, as stated explicitly in the paper's abstract.
                """,
                """
    This paper belongs to the "Genetic Algorithms" category because it is focused on providing a user's guide for the "PGAPack Parallel Genetic Algorithm Library", which directly implies that the main subject of the paper is related to genetic algorithms.
                """,
                """
    This paper belongs to the "Theory" category because it examines abstract concepts like optimality, domination, bounded convergence rates, and the idea of a "grace period" in the context of repeated stage games, all of which are theoretical considerations that do not pertain specifically to any of the other given categories such as rule learning, neural networks, case based, genetic algorithms, reinforcement learning, or probabilistic methods.
                """,
                """
    The paper belongs to the "Reinforcement Learning" category because it presents experiments involving neural networks and genetic algorithms, and notably finds that reinforcement learning, a key focus in the study, provides a significant advantage when used with a specific type of sensor, as detailed in the findings and analysis of the research study.
                """,
                """
    This paper belongs to the "Probabilistic Methods" category because it focuses on the study of Bayesian networks, a key tool in probabilistic methods, utilizing nonlinear conditional density estimators and belief update sampling techniques, which are all concepts deeply rooted in probabilistic theory and methodology.
                """
            ],
            "citeseer": [
                """
                This paper belongs to the "Database" category because it introduces the concept of treating data streams as relational database tables, uses SQL-like queries to retrieve information from these streams, incorporates computations and transformations akin to operations on database views, optimizes these queries, and presents a data management system, dQUOB, based on these database-centric principles.            """,
                """
                This paper belongs to the category of "Machine Learning" because it extensively discusses pattern recognition which is a fundamental concept in machine learning, focuses on statistical learning theory which is a cornerstone of many machine learning algorithms, and explores topics such as feature extraction and selection, classifier design and learning, and performance evaluation, all of which are key steps in the machine learning process.
                """,
                """
                This paper belongs to the category of "Human Computer Interaction" because it discusses usability evaluations, a key topic in HCI, specifically focusing on the analysis and improvement of the user experience in groupware applications, which involves understanding human behavior, interaction mechanisms, and the context of use, all fundamental elements of Human Computer Interaction.
                """,
                """
                This paper belongs to the "Agents" category because it discusses "Emergent Cooperative Goal-Satisfaction in Large Scale Automated-Agent Systems," specifically focusing on autonomous agents' cooperation in large-scale multi-agent systems, incorporating approaches related to cooperation, negotiation, distributed planning, and coalition formation, all central topics in the study of agent systems.
                """,
                """
                This paper belongs to the category of "Artificial Intelligence" because it discusses the design and implementation of an autonomous mobile robot, a subject central to AI, and explores advanced AI concepts such as managing the robot's goals with internal variables or 'motives', enabling it to adapt to changing environments, and facilitating its interaction with other agents through a visual communication system.
                """,
                """
                This paper belongs to the category of "Information Retrieval" because it discusses methods such as the Kleinberg HITS and the Google PageRank algorithms, which are foundational techniques in the field of Information Retrieval used for analyzing hyperlink or citation data to identify authoritative or influential resources in a network of data, thus facilitating the retrieval of the most relevant information.
                """
            ],
            "arxiv": ["""
            It discusses theoretical problems related to data types, and presents a solution ("exp-log normal form of types") for type isomorphism, a concept in the logic in computer science. Therefore, Logic in Computer Science (cs.LO) seems suitable.
            The usage of the Coq proof assistant for implementation further suggests a link to Logic in Computer Science (cs.LO), as Coq is often used in formal methods and logic research.
            """, 
            """
            The paper is about the Ubuntu Dialogue Corpus, a large dataset suitable for natural language processing and computational linguistics research, which places it in the cs.CL category.
            """,
            """
            First, I identified the core topics and technical terms in the paper, such as PNC, ANC, and soft-input soft-output decoding.
            Then, I matched these topics with corresponding arXiv subcategories. For instance, the concepts of PNC and ANC fall under the Information Theory category (cs.IT), given their focus on the efficient communication of information.
            """,
            """
            The paper presents algorithmic developments, such as a polynomial-time algorithm for the k-means clustering problem and an efficient (1+\eps)-approximation algorithm. The development and analysis of such algorithms fall under the category of data structures and algorithms.
            """,
            """
            Considering the focus on permutations, algorithmic complexity, use of discrete data structures (like stacks), and the problem-solving nature of the paper, it can be classified under the category of Discrete Mathematics in arxiv.
            """,
            """
            This paper belongs to the arXiv category of Computation and Language (cs.CL) because it focuses on the computational analysis of language, specifically developing an unsupervised method for identifying changes in word senses over time, using data-driven techniques on large text corpora, a topic that lies at the intersection of natural language processing, computational linguistics, and machine learning, all of which are fundamental areas within cs.CL.
            """
            ],
            "products": [
                """
                This product belongs to the "Books" category because the description mentions that it is a "Little Book of Whittling," indicating that the product is a physical book and thus aligns with the classification of a product in the "Books" category.
                """, 
                """
                This product, the Zojirushi BB-CEC20 Home Bakery Supreme 2-Pound-Loaf Breadmaker, belongs to the "Home & Kitchen" category because it is a breadmaker, which is a common kitchen appliance used in homes for baking bread.
                """,
                """
                This product belongs to the "Pet Supplies" category because it is described as a "Self-Grooming Aid for Cats," which indicates that it is a product specifically designed for the care and maintenance of pets, in this case, cats.
                """,
                """
                This product, "Columbia Women's Silver Ridge Plaid Long Sleeve Shirt," belongs to the 'Sports & Outdoors' category because its description mentions built-in UPF 40 sun block and its suitability for active days outdoors, indicating that it is specifically designed for outdoor sporting and recreational activities.
                """,
                """
                This product belongs to the 'Toys & Games' category because the description explicitly refers to it as a toy, mentions play, and highlights features such as hand-painted design and life-like modeling, all attributes typically associated with items classified under toys and games.
                """,
                """
                This product, the CS Hyde Metalized Mylar Tape, belongs to the 'Tools & Home Improvement' category because it is a versatile tape with specific functional properties such as excellent solvent resistance, heat reflectivity, and high adhesive strength, all of which are typically required in various DIY, repair, maintenance, or improvement tasks at home or in professional settings.
                """
            ],
            "pubmed": [
                """
                This paper belongs to the category "Diabetes Mellitus Type 2" because it specifically investigates the glycemic response in type 2 diabetic patients to different antihyperglycemic therapies such as metformin, sulfonylureas, thiazolidinediones, and insulin, and suggests the need for aggressive therapy intensification for better glycemic control, which is characteristic of type 2 diabetes management.
                """,
                """
                This paper belongs to the "Diabetes Mellitus Type 1" category because it specifically investigates thyroid autoimmunity in a large cohort of children and adolescents with type 1 diabetes, as clearly stated in the objective of the abstract.
                """,
                """
                This paper belongs to the "Diabetes Mellitus, Experimental" category because it investigates the effects of experimentally induced diabetes on somatostatin mRNA expression in the rat brain, with the diabetes being induced in the rat models by the injection of streptozotocin, a common experimental method for inducing diabetes in research studies.
                """
            ]
        }
    }
    return few_shot_samples