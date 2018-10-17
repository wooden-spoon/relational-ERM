"""
Minimal script to create a tensorflow dataset from graphsage style data and draw a sample from it

"""
import sys
from collections import namedtuple

import tensorflow as tf
import numpy as np

from scripts.dataset_logic import load_data_graphsage
from scripts.run_skipgram_simple import parse_arguments, \
    _adjust_regularization, _adjust_learning_rate, _make_global_optimizer

from relational_sgd.sampling import adapters, factories

from relational_sgd.models.skipgram import make_class_prediction_with_features



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

        if args.batch_size is not None:
            dataset = dataset.apply(
                adapters.padded_batch_samples(args.batch_size))

        dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

        return dataset

    return input_fn


def main():
    tf.enable_eager_execution()

    # create fake args for debugging
    sys.argv = ['']
    args = parse_arguments()

    graph_data = load_data_graphsage()


    """
    Sample defined as a custom tf dataset
    """
    sample_train = get_dataset_fn(args.sampler, args)

    """
    Predictor class and loss function
    """
    # hyperparams
    args.label_task_weight = 1e-3  # balance of graph edge prediction vs label prediction loss
    args.global_regularization = 1.  # regularization for global params

    classifier_predictor_and_loss = make_class_prediction_with_features(
        label_task_weight=args.label_task_weight,
        regularization=_adjust_regularization(args.global_regularization, args.batch_size),
        global_optimizer=_make_global_optimizer(args),
        embedding_optimizer=lambda: tf.train.GradientDescentOptimizer(
            _adjust_learning_rate(args.embedding_learning_rate, args.batch_size)),
        polyak=False)

    vertex_embedding_params = {
        'embedding_dim': args.embedding_dim,
        'embedding_trainable': True,
        'embedding_checkpoint': None
    }

    params={
        **vertex_embedding_params,
        'hidden_units' : [200, 200]  # Jaan net,
        'n_classes':  max(graph_data.classes)+1,
        'num_vertices': graph_data.num_vertices,
        'batch_size': args.batch_size
    }

    node_classifier = tf.estimator.Estimator(
        model_fn=classifier_predictor_and_loss,
        params=params,
        model_dir=args.train_dir)


    """
    Put it together for the optimization
    """
    node_classifier.train(
        input_fn=make_input_fn(graph_data, args, sample_train),
        max_steps=args.max_steps)

