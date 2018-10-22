import warnings

import numpy as np
import tensorflow as tf

from collections import namedtuple
from relational_erm.graph_ops.representations import PackedAdjacencyList

GraphDataGS = namedtuple('GraphDataGS', ['edge_list',
                                         'node_features',
                                         'classes',
                                         'adjacency_list',
                                         'num_vertices'])


def load_data_graphsage(data_path=None):
    """
    Loads preprocessed data stores as .npz with keys
    ['edge_list', 'features', 'classes', 'node_ids', 'neighbours', 'lengths']

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

    num_vertices = len(lengths)

    # vertex attributes
    classes = loaded['classes'].astype(np.int32)
    node_features = loaded['features'].astype(np.float32)

    return GraphDataGS(edge_list=edge_list,
                       classes=classes,
                       node_features=node_features,
                       adjacency_list=adjacency_list,
                       num_vertices=num_vertices)
