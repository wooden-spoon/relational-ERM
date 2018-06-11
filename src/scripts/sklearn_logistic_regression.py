"""
Load pretrained embeddings and use the node2vec testing procedure to
1. train logistic regression to predict vertex labels from embeddings
2. score using DeepWalk's scoring code (to match w node2vec)
"""

import os
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score

import tensorflow as tf

from models.scoring import TopKRanker
from relational_sgd.data_processing.load_erm_embeddings import process_erm_emb


def logistic_reg(embeddings, labels, seed):
    rng = np.random.RandomState(seed)
    in_train = rng.binomial(1, 0.5, embeddings.shape[0]).astype(np.bool)
    in_test = np.logical_not(in_train)

    # classifier
    classifier = TopKRanker(linear_model.LogisticRegression(C=1.))
    classifier.fit(X=embeddings[in_train], y=labels.astype(np.int)[in_train])

    top_k_list = list(np.sum(labels[in_test], 1).astype(np.int))
    pred_labels = classifier.predict(embeddings[in_test], top_k_list)
    mlb = MultiLabelBinarizer(range(labels.shape[1]))
    pred_labels = mlb.fit_transform(pred_labels)

    f1_macro = f1_score(labels[in_test], pred_labels, average='macro')
    f1_micro = f1_score(labels[in_test], pred_labels, average='micro')

    return f1_macro, f1_micro


def get_dataset_info(dataset):
    # protein-protein
    if dataset == 'homo-sapiens':
        # data
        data_dir = '../data/homo_sapiens'
        data_file = 'homo_sapiens.npz'
        data_path = os.path.join(data_dir, data_file)
        with np.load(data_path, allow_pickle=False) as loaded:
            labels = loaded['group'].astype(np.float32)

        num_vertices = 3852

    elif dataset == 'wikipedia_word_coocurr':
        # data
        data_dir = '../data/wikipedia_word_coocurr'
        data_file = 'wiki_pos.npz'
        data_path = os.path.join(data_dir, data_file)
        with np.load(data_path, allow_pickle=False) as loaded:
            labels = loaded['group'].astype(np.float32)

        num_vertices = 4777

    elif dataset == 'blog_catalog':
        # data
        data_dir = '../data/blog_catalog_3'
        data_file = 'blog_catalog.npz'
        data_path = os.path.join(data_dir, data_file)
        with np.load(data_path, allow_pickle=False) as loaded:
            labels = loaded['group'].astype(np.float32)

        num_vertices = 10312
    else:
        raise Exception("Unrecognized dataset!")

    return labels, num_vertices


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding-dir', type=str)
    parser.add_argument('--dataset', type=str, default='homo-sapiens')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    labels, num_vertices = get_dataset_info(args.dataset)

    ckpt_path = tf.train.latest_checkpoint(args.embedding_dir)
    np_embeddings = process_erm_emb(
        ckpt_path, num_vertices,
        tensor_name_in_ckpt="input_layer/vertex_index_embedding/embedding_weights",
        embedding_dim=128)
    f1_macro, f1_micro = logistic_reg(np_embeddings, labels, args.seed)
    print("f1_macro: {}".format(f1_macro))
    print("f1_micro: {}".format(f1_micro))


if __name__ == '__main__':
    main()
