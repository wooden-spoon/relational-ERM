"""
Load and process wikipedia word cooccurrance data

data from http://snap.stanford.edu/node2vec/#datasets
"""

import scipy.io
import os
import numpy as np


def main():
    wiki_dir = '../data/wikipedia_word_coocurr'
    wiki_file = os.path.join(wiki_dir, 'POS.mat')

    mat = scipy.io.loadmat(wiki_file)

    group = mat['group'].toarray()
    network = mat['network']

    nnz = network.nonzero()
    edge_list = np.stack(nnz, axis=1)
    edge_list_undirected = (edge_list[:,0] <= edge_list[:,1])
    edge_list = edge_list[edge_list_undirected]

    weights = network[network.nonzero()].T
    weights = weights[edge_list_undirected]

    save_path = os.path.join(wiki_dir, 'wiki_pos.npz')
    np.savez_compressed(save_path, edge_list=edge_list, weights=weights, group=group)


if __name__ == '__main__':
    main()
