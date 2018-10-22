import os
import numpy as np
import tensorflow as tf

from relational_erm.graph_ops.representations import edge_list_to_adj_list, create_packed_adjacency_list
from relational_erm.tensorflow_ops import adapter_ops as ops

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def create_graph(num_edges, num_vertex):
    rng = np.random.RandomState(42)
    edges_start = rng.randint(0, num_vertex, size=num_edges)
    edges_offset = rng.randint(1, num_vertex, size=num_edges)
    edges_end = np.mod(edges_start + edges_offset, num_vertex)
    edges = np.stack((edges_start, edges_end), axis=1)
    edges.sort(axis=1)
    edges = np.unique(edges, axis=0)
    _, edges_unique = np.unique(edges, return_inverse=True)
    edges = np.reshape(edges_unique, edges.shape)
    adj_list = create_packed_adjacency_list(edge_list_to_adj_list(edges, np.ones(edges.shape[0])))
    return adj_list, edges


def test_adapt_edge_list():
    adj_list, edges = create_graph(20, 10)

    with tf.Graph().as_default():
        edge_list_op = ops.adjacency_to_edge_list(adj_list.neighbours, adj_list.lengths, redundant=False)

        with tf.Session() as session:
            new_edge_list = session.run(edge_list_op)

    assert new_edge_list.shape == edges.shape


def test_adapt_edge_list_posneg_no_redundant():
    adj_list, edges = create_graph(20, 10)

    with tf.Graph().as_default():
        edge_list_op = ops.adjacency_to_posneg_edge_list(adj_list.neighbours, adj_list.lengths, redundant=False)

        with tf.Session() as session:
            new_edge_list_pos, new_edge_list_neg = session.run(edge_list_op)

    assert new_edge_list_pos.shape[0] == edges.shape[0]


def test_adapt_edge_list_posneg_redundant():
    adj_list, edges = create_graph(20, 10)

    with tf.Graph().as_default():
        edge_list_op = ops.adjacency_to_posneg_edge_list(adj_list.neighbours, adj_list.lengths, redundant=True)

        with tf.Session() as session:
            new_edge_list_pos, new_edge_list_neg = session.run(edge_list_op)

    assert new_edge_list_pos.shape[0] == edges.shape[0] * 2


def test_get_induced_subgraph():
    adj_list, edges = create_graph(20, 10)

    with tf.Graph().as_default():
        subgraph_op = ops.get_induced_subgraph(
            [1, 3, 5], adj_list.neighbours, adj_list.lengths, adj_list.offsets)

        with tf.Session() as session:
            neighbours, lengths, offsets = session.run(subgraph_op)

    assert len(lengths) == 3
    assert len(offsets) == 3


def test_open_ego_network():
    adj_list, edges = create_graph(20, 10)

    with tf.Graph().as_default():
        ego_edges_op = ops.get_open_ego_network(np.array([1, 3, 5], dtype=np.int32),
                                                adj_list.neighbours, adj_list.lengths, adj_list.offsets)

        with tf.Session() as session:
            ego_edges = session.run(ego_edges_op)

    assert ego_edges.shape[1] == 2
    assert ego_edges.shape[0] == adj_list.lengths[[1, 3, 5]].sum()
