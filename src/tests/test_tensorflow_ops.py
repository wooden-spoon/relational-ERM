import os
import pytest

import relational_erm.tensorflow_ops.array_ops
from relational_erm.graph_ops.representations import edge_list_to_adj_list, create_packed_adjacency_list
from relational_erm.tensorflow_ops import dataset_ops as ops
from relational_erm.tensorflow_ops import adapter_ops
import numpy as np
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def load_data_adjacency_list():
    data_dir = '../data/homo_sapiens'

    data_path = os.path.join(data_dir, 'homo_sapiens.npz')

    loaded = np.load(data_path)
    edge_list = loaded['edge_list']
    weights = loaded['weights']
    labels = loaded['group']

    adjacency_list = edge_list_to_adj_list(edge_list, weights)
    adjacency_list = create_packed_adjacency_list(adjacency_list)
    return adjacency_list, labels


def test_biased_walk_tensorflow():
    adj_list, labels = load_data_adjacency_list()

    dataset = ops.BiasedRandomWalkDataset(
        100, 1, 1, adj_list.neighbours, adj_list.lengths, adj_list.offsets)

    n_value = next(iter(dataset))

    assert 'walk' in n_value
    assert len(n_value['walk']) == 100


def test_biased_walk_tensorflow_shape_validation():
    adj_list, labels = load_data_adjacency_list()

    with pytest.raises(ValueError):
        with tf.Graph().as_default():
            dataset = ops.BiasedRandomWalkDataset(
                100, np.array([1, 2], dtype=np.float32), 1, adj_list.neighbours, adj_list.lengths, adj_list.offsets)

            it = dataset.make_one_shot_iterator()

            n = it.get_next()

            with tf.Session() as session:
                _ = session.run(n)


def test_uniform_edge_sampler():
    adj_list, labels = load_data_adjacency_list()

    dataset = ops.UniformEdgeDataset(100, adj_list.neighbours, adj_list.lengths, adj_list.offsets)
    n_value = next(iter(dataset))

    assert n_value['edge_list'].shape == (100, 2)


def test_random_walk_sampler():
    adj_list, labels = load_data_adjacency_list()

    dataset = ops.RandomWalkDataset(20, adj_list.neighbours, adj_list.lengths, adj_list.offsets)

    n_value = next(iter(dataset))

    assert n_value['walk'].shape[0] == 20
