"""
Load and process blog catalog 3 data

data from http://socialcomputing.asu.edu/datasets/BlogCatalog3
"""

import pandas as pd
import numpy as np
import scipy as sp
import networkx as nx
import os


def _nx_from_edgelist(edge_list):
    V = np.unique(edge_list).shape[0]
    as_csr = sp.sparse.csr_matrix((np.ones_like(edge_list[:, 0]), (edge_list[:, 0], edge_list[:, 1])), [V,V])
    G = nx.from_scipy_sparse_matrix(as_csr)
    return G


def main():
    bc_dir = '../data/blog_catalog_3'
    el_file = os.path.join(bc_dir, 'edges.csv')
    groups_file = os.path.join(bc_dir, 'group-edges.csv')

    edge_list = pd.read_csv(el_file)
    edge_list = edge_list.as_matrix()

    groups = pd.read_csv(groups_file, names=['vert_id', 'label'])

    # index from 0
    edge_list = edge_list - 1
    groups['vert_id'] = groups['vert_id'] - 1
    groups['label'] = groups['label'] - 1

    # labels in one-hot format
    n_vert = np.unique(edge_list).shape[0]
    n_labels = groups['label'].max() + 1
    labels = np.zeros([n_vert, n_labels], dtype=np.int32)
    labels[groups['vert_id'], groups['label']] = 1

    # simple graph (so weights all 1)
    weights = np.ones(edge_list.shape[0], dtype=np.float32)

    # save
    save_path = os.path.join(bc_dir, 'blog_catalog.npz')
    np.savez_compressed(save_path, edge_list=edge_list, weights=weights, group=labels)

    # edge list csv to pass to node2vec
    nx.write_edgelist(_nx_from_edgelist(edge_list), os.path.join(bc_dir, "blog_catalog.edgelist"))

if __name__ == '__main__':
    main()
