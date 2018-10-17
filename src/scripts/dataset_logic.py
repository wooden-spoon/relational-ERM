import warnings

import numpy as np
import tensorflow as tf

from collections import namedtuple

from relational_sgd.graph_ops.representations import create_packed_adjacency_list, edge_list_to_adj_list
from relational_sgd.graph_ops.representations import PackedAdjacencyList

GraphDataGS = namedtuple('GraphDataGS', ['edge_list',
                                         'node_features',
                                         'classes',
                                         'adjacency_list',
                                         'num_vertices'])


def load_data_graphsage(data_path=None):
    """
    Loads preprocessed data stores as .npz with keys
    ['edge_list', 'weights', 'features', 'classes', 'node_ids', 'neighbours', 'lengths']

    'features' is a real-valued array of shape [n_vert, feature_dimension] giving
    some (embedding of) features for each vertex

    'classes' are integers representing category labels of the vertices

    Parameters
    ----------
    data_path: path to the file to load

    Returns
    -------
    An instance of GraphDataN2V containing the parsed graph data for the dataset.

    """
    if data_path is None:
        data_path = '../data/reddit/reddit.npz'

    # use tensorflow loading to support loading from
    # cloud providers
    with tf.gfile.Open(data_path, mode='rb') as f:
        loaded = np.load(f, allow_pickle=False)

    # graph attributes
    edge_list = loaded['edge_list'].astype(np.int32)

    if 'weights' in loaded:
        warnings.warn("edge weights are not implemented for graphsage loader and will be ignored!")
        # weights = loaded['weights'].astype(np.float32)

    weights = None

    neighbours = loaded['neighbours']
    lengths = loaded['lengths']

    offsets = np.empty_like(lengths)
    np.cumsum(lengths[:-1], out=offsets[1:])
    offsets[0] = 0

    adjacency_list = PackedAdjacencyList(neighbours, None, offsets, lengths, np.arange(len(lengths)))

    num_vertices = len(neighbours)

    # vertex attributes
    classes = loaded['classes'].astype(np.int32)
    node_features = loaded['features'].astype(np.float32)

    return GraphDataGS(edge_list=edge_list,
                       classes=classes,
                       node_features=node_features,
                       adjacency_list=adjacency_list,
                       num_vertices=num_vertices)


GraphDataN2V = namedtuple('GraphDataN2V', ['edge_list',
                                           'weights',
                                           'labels',
                                           'adjacency_list',
                                           'num_vertices',
                                           'num_labels'])


def load_data_node2vec(data_path=None):
    """ Loads any of
    1. homo sapiens
    2. wikipedia parts of speech
    3. blog catalog 3
    assumed to be preprocessed as undirected graphs according to scripts in data_processing

    Parameters
    ----------
    data_path: The path to the node 2 vec data file to load.

    Returns
    -------
    An instance of GraphDataN2V containing the parsed graph data for the dataset.
    """
    if data_path is None:
        data_path = '../data/homo_sapiens/homo_sapiens.npz'

    # data_path = '../data/blog_catalog_3/blog_catalog.npz'
    # data_path = '../../data/wikipedia_word_coocurr/wiki_pos.npz'

    # use tensorflow loading to support loading from
    # cloud providers
    with tf.gfile.Open(data_path, mode='rb') as f:
        loaded = np.load(f, allow_pickle=False)

    edge_list = loaded['edge_list'].astype(np.int32)

    if 'weights' in loaded:
        weights = loaded['weights'].astype(np.float32)
    else:
        weights = np.ones(edge_list.shape[0], dtype=np.float32)

    labels = loaded['group'].astype(np.float32)

    # Remove self-edges
    not_self_edge = edge_list[:, 0] != edge_list[:, 1]
    edge_list = edge_list[not_self_edge, :]
    weights = weights[not_self_edge]

    adjacency_list = edge_list_to_adj_list(edge_list, weights)

    num_vertices = len(adjacency_list)
    adjacency_list = create_packed_adjacency_list(adjacency_list)
    num_labels = labels.shape[1]

    return GraphDataN2V(edge_list=edge_list,
                        weights=weights,
                        labels=labels,
                        adjacency_list=adjacency_list,
                        num_vertices=num_vertices,
                        num_labels=num_labels)


def load_data_wikipedia_hyperlink(data_path=None):
    """ Load the wikipedia hyperlink data.

    Parameters
    ----------
    data_path: the path to the preprocessed dataset.
    """
    if data_path is None:
        data_path = '../data/wikipedia_hlink/wikipedia_hlink_preprocessed.npz'

    with tf.gfile.Open(data_path, mode='rb') as f:
        loaded = np.load(f, allow_pickle=True)

    neighbours = loaded['neighbours']
    lengths = loaded['lengths']

    offsets = np.empty_like(lengths)
    np.cumsum(lengths[:-1], out=offsets[1:])
    offsets[0] = 0

    adjacency_list = PackedAdjacencyList(neighbours, None, offsets, lengths, np.arange(len(lengths)))
    labels_sparse = loaded['sparse_labels'].astype(np.int32, copy=False)

    return {
        'adjacency_list': adjacency_list,
        'labels_sparse': labels_sparse
    }
