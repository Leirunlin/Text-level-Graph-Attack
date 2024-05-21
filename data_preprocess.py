import os
import pickle
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sentence_transformers import SentenceTransformer
import numpy as np


def get_tf_idf_by_texts(texts, max_features=500, known_mask=None):
    if known_mask == None:
        tf_idf_vec = TfidfVectorizer(stop_words='english', max_features=max_features)
        X = tf_idf_vec.fit_transform(texts)
        torch_feat = torch.FloatTensor(X.todense())
        norm_torch_feat = F.normalize(torch_feat, dim = -1)
    else:
        tf_idf_vec = TfidfVectorizer(stop_words='english', max_features=max_features)
        text_known = texts[known_mask]
        text_test = texts[~known_mask]
        x_known = tf_idf_vec.fit_transform(text_known)
        x_test = tf_idf_vec.transform(text_test)
        x_known = torch.FloatTensor(x_known.todense())
        x_test = torch.FloatTensor(x_test.todense())
        dim = x_known.shape[1]
        torch_feat = torch.zeros(len(texts), dim)
        torch_feat[known_mask] = x_known 
        torch_feat[~known_mask] = x_test
        norm_torch_feat = F.normalize(torch_feat, dim = -1)
    return torch_feat, norm_torch_feat


def save_vectorizer(vec, filename):
    with open(filename, 'wb') as f:
        pickle.dump(vec, f)


def load_vectorizer(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def get_bow_by_texts(texts, dataset, max_features=500, all=False):
    vectorizer_path = os.path.join("./bow_cache/", f"{dataset}.pkl")
    if not all:
        # Use default bow vocabulary
        if vectorizer_path and os.path.exists(vectorizer_path):
            print("Loading bow .pkl")
            vec = load_vectorizer(vectorizer_path)
            X = vec.transform(texts)
        else:
            vec = CountVectorizer(max_features=max_features, stop_words='english', binary=True)
            X = vec.fit_transform(texts)
            if vectorizer_path:
                save_vectorizer(vec, vectorizer_path)
    else:
        # Use all text as vocabulary
        vec = CountVectorizer(max_features=max_features, stop_words='english', binary=True)
        X = vec.fit_transform(texts)

    torch_feat = torch.FloatTensor(X.toarray())
    norm_torch_feat = F.normalize(torch_feat, p=2, dim=-1)
    return torch_feat, norm_torch_feat


def get_sbert_emb(texts, device):
    model = SentenceTransformer().to(device)
    sbert_embeds = model.encode(texts, batch_size=32, show_progress_bar=True)
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


def text2emb(texts, dataset, embdding='bow', all=False):
    if embdding in ['bow','bow_all']:
        x, norm_x = get_bow_by_texts(texts, dataset, all=all)
    elif embdding == 'sbert':
        x = get_sbert_emb(texts, device="cuda")
    elif embdding == 'tfidf':
        x, norm_x = get_tf_idf_by_texts(texts)
    elif embdding == 'gtr':
        x = get_gtr_emb(texts)
    return x


def text_stats(texts):
    word_counts = [len(text.split()) for text in texts]
    max_words = max(word_counts)
    min_words = min(word_counts)
    average_words = np.mean(word_counts)
    std_deviation = np.std(word_counts)
    
    return max_words, min_words, average_words, std_deviation