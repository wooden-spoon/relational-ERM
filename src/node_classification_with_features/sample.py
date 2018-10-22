"""
Example illustrating how to create Sample for a graph dataset where each vertex includes a real-valued vertex
attribute
"""

import tensorflow as tf
import numpy as np

from relational_erm.sampling import adapters, factories

from collections import namedtuple
from node_classification_with_features.dataset_logic.dataset_logic import load_data_graphsage

def make_sample(sampler, args):
    if sampler is None:
        sampler = 'biased-walk'

    return factories.make_dataset(sampler, args)


def augment_sample(graph_data, args, dataset_fn=None, num_samples=None):
    """
    Augments a graph sampler by (i) standardizing subgraph formatting,
    and (ii) adding node features and classes

    Parameters
    ----------
    graph_data: graph data object includes all vertex feature and label information
    args: parsed arguments, includes optimization and testing parameters
    dataset_fn: graph sampling algorithm
    num_samples: if specified, only num_samples subgraphs are returned (useful for test time)

    Returns
    -------
    tensorflow dataset that generates subgraphs augmented with node features and classes
    """
    def input_fn():
        dataset = dataset_fn(graph_data, args.seed)

        data_processing = adapters.compose(
            adapters.relabel_subgraph(),
            adapters.append_vertex_vector_features(graph_data.node_features),
            adapters.append_sparse_vertex_classes(graph_data.classes),
            adapters.split_vertex_labels(
                graph_data.num_vertices, args.proportion_censored,
                np.random.RandomState(args.seed)),
            adapters.add_sample_size_info(),
            adapters.format_features_labels())

        dataset = dataset.map(data_processing, 4)
        if num_samples is not None:
            dataset = dataset.take(num_samples)

        if args.batch_size is not None:
            dataset = dataset.apply(
                adapters.padded_batch_samples(args.batch_size))

        dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

        return dataset

    return input_fn


def _create_dev_dataset(graph_data):
    from relational_erm.graph_ops.representations import create_packed_adjacency_from_redundant_edge_list
    """
    Use the sampler to create a small version of a full dataset for fast development

    Parameters
    ----------
    graph_data

    Returns
    -------

    """
    args = namedtuple('dummyArgs', ['sampler',
                                    'num_edges',
                                    'window_size',
                                    'dataset_shards',
                                    'num_negative',
                                    'num_negative_total',
                                    'proportion_censored',
                                    'seed',
                                    'batch_size'])

    args.sampler = 'random-walk'
    args.num_edges = 2000
    args.window_size = 1
    args.dataset_shards = 1
    args.num_negative = 5
    args.num_negative_total = None
    args.proportion_censored = 0.5
    args.seed = 0
    args.batch_size = 1

    dataset_fn = factories.make_biased_walk_induced_dataset(args)
    input_fn = augment_sample(graph_data, args, dataset_fn)

    dataset = input_fn()
    sampler = dataset.make_one_shot_iterator()
    (features, labels) = sampler.get_next()

    edge_list = features['edge_list'].numpy()
    vertex_features = features['vertex_features'].numpy()
    classes = labels['labels'].numpy()

    # produce stuff that loader expects

    red_edge_list = np.concatenate((edge_list, np.flip(edge_list, axis=1)))
    packed_adj_list = create_packed_adjacency_from_redundant_edge_list(red_edge_list )
    neighbours = packed_adj_list.neighbours
    lengths = packed_adj_list.lengths

    dev_dataset = {'edge_list': edge_list,
                   'features': vertex_features,
                   'classes': classes,
                   'neighbours': neighbours,
                   'lengths': lengths}

    np.savez_compressed('reddit-dev.npz', **dev_dataset)


def main():
    tf.enable_eager_execution()

    graph_data = load_data_graphsage('../data/reddit/reddit-dev.npz')

    args = namedtuple('dummyArgs', ['sampler',
                                    'num_edges',
                                    'window_size',
                                    'dataset_shards',
                                    'num_negative',
                                    'num_negative_total',
                                    'proportion_censored',
                                    'seed',
                                    'batch_size'])

    args.sampler = 'random-walk'
    args.num_edges = 800
    args.window_size = 10
    args.dataset_shards = 1
    args.num_negative = 5
    args.num_negative_total = None
    args.proportion_censored = 0.5
    args.seed = 0
    args.batch_size = 10

    # testing dataset
    sample = make_sample(args.sampler, args)
    input_fn = augment_sample(graph_data, args, sample)

    dataset = input_fn()
    ds_itr = dataset.make_one_shot_iterator()
    (features, labels)=ds_itr.get_next()