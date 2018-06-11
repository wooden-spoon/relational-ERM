"""
Load and process wikipedia hyperlink network

Data includes:
hyperlink structure
categories for each article
article name

data from http://snap.stanford.edu/data/wiki-topcats.html
"""

import os
import numpy as np
import gzip


def _load_article_names(name_file):
    file_names = []

    with gzip.open(name_file, 'rt') as f:
        for line in f:
            file_names += [" ".join(line.split()[1:])]

    return file_names


def _load_article_topics(topics_file):
    categories = []
    category_articles = []

    with gzip.open(topics_file, 'rt') as f:
        for i, line in enumerate(f):
            category, articles = line.split(';')
            categories.append(category.split(':')[1])
            articles = np.fromstring(articles.strip(), dtype=np.int32, sep=' ')

            category_articles.append(np.stack([np.ones_like(articles) * i, articles], axis=1))

    labels_indices = np.concatenate(category_articles, axis=0)

    return categories, labels_indices


def preprocess_data(data_directory='../data/wikipedia_hlink'):
    link_file = os.path.join(data_directory, 'wiki-topcats.txt.gz')
    name_file = os.path.join(data_directory, 'wiki-topcats-page-names.txt.gz')
    category_file = os.path.join(data_directory, 'wiki-topcats-categories.txt.gz')

    edge_list = np.loadtxt(link_file, dtype=np.int32)
    categories, article_categories = _load_article_topics(category_file)
    article_names = _load_article_names(name_file)

    return {
        'edge_list': edge_list,
        'categories': categories,
        'labels': article_categories,
        'article_names': article_names
    }


def preprocess_packed_adjacency_list(data):
    from ..graph_ops.representations import create_packed_adjacency_from_edge_list

    # Load the current edge list, and go to canonical form
    # without self edges
    edge_list = data['edge_list']
    edge_list = edge_list[edge_list[:, 0] != edge_list[:, 1], :]
    edge_list.sort(axis=-1)
    edge_list = np.unique(edge_list, axis=0)

    # Compute redundant edge list
    edge_list = np.concatenate((edge_list, np.flip(edge_list, axis=1)))
    packed_adj_list = create_packed_adjacency_from_edge_list(edge_list)

    return {
        'neighbours': packed_adj_list.neighbours,
        'lengths': packed_adj_list.lengths,
        'sparse_labels': data['labels']
    }


def main():
    data = preprocess_data()
    np.savez_compressed('wikipedia_hlink.npz', **data)

    data = preprocess_packed_adjacency_list(data)
    np.savez_compressed('wikipedia_hlink_processed.npz', **data)


if __name__ == '__main__':
    main()

