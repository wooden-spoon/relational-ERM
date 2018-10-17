"""
Minimal script to create a tensorflow dataset from graphsage style data and draw a sample from it
"""

import tensorflow as tf
import numpy as np

from relational_sgd.sampling import adapters, factories

from scripts.dataset_logic import load_data_graphsage

from collections import namedtuple


def get_dataset_fn(sampler, args):
    if sampler is None:
        sampler = 'biased-walk'

    return factories.make_dataset(sampler, args)


def make_input_fn(graph_data, args, dataset_fn=None, num_samples=None):
    def input_fn():
        dataset = dataset_fn(graph_data, args.seed)

        data_processing = adapters.compose(
            adapters.relabel_subgraph(),
            adapters.append_vertex_vector_features(graph_data.node_features),
            adapters.append_vertex_classes(graph_data.classes),
            adapters.split_vertex_labels(
                graph_data.num_vertices, args.proportion_censored,
                np.random.RandomState(args.seed)),
            adapters.add_sample_size_info(),
            adapters.format_features_labels())

        dataset = dataset.map(data_processing, 4)
        if num_samples is not None:
            dataset = dataset.take(num_samples)

        # if args.batch_size is not None:
        #     dataset = dataset.apply(
        #         adapters.padded_batch_samples(args.batch_size))

        dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

        return dataset

    return input_fn


def main():
    tf.enable_eager_execution()

    graph_data = load_data_graphsage()

    args = namedtuple('dummyArgs', ['num_edges',
                                    'window_size',
                                    'dataset_shards',
                                    'num_negative',
                                    'num_negative_total',
                                    'proportion_censored',
                                    'seed',
                                    'batch_size'])

    args.num_edges = 800
    args.window_size = 10
    args.dataset_shards = 1
    args.num_negative = 5
    args.num_negative_total = None
    args.proportion_censored = 0.5
    args.seed = 0
    args.batch_size = 10

    # testing dataset
    dataset_fn = factories.make_biased_random_walk_dataset(args)
    # dataset = dataset_fn(graph_data, args.seed)
    input_fn = make_input_fn(graph_data, args, dataset_fn)

    dataset = input_fn()
    # sampler = dataset.make_one_shot_iterator()
    # samp = sampler.get_next()

    dataset = dataset.apply(
        adapters.padded_batch_samples(args.batch_size))

    ds_itr = dataset.make_one_shot_iterator()
    (features, labels)=ds_itr.get_next()

