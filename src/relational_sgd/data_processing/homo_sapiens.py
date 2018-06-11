"""
Load and process homo-sapien protein-protein interaction data

We treat the network as undirected

data from http://snap.stanford.edu/node2vec/#datasets
"""

import scipy.io
import os
import numpy as np
import networkx as nx
from relational_sgd.graph_ops.representations import relabel


def main():
    hs_dir = '../data/homo_sapiens'
    hs_file = os.path.join(hs_dir, 'Homo_sapiens.mat')

    mat = scipy.io.loadmat(hs_file)

    group = mat['group'].toarray()
    network = mat['network']

    ppi = nx.from_scipy_sparse_matrix(network)
    largest_cc = max(nx.connected_components(ppi), key=len)
    connected_ppi = ppi.subgraph(largest_cc)

    verts_in_largest_cc = np.array(list(largest_cc))

    edge_list = np.array(list(connected_ppi.edges()))

    relab_edge_list, vert_index = relabel(edge_list)

    weights = np.ones([relab_edge_list.shape[0], 1])
    group = group[verts_in_largest_cc]

    save_path = os.path.join(hs_dir, 'homo_sapiens.npz')
    np.savez_compressed(save_path, edge_list=relab_edge_list, weights=weights, group=group)


if __name__ == '__main__':
    main()
