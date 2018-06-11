""" This script implements a simple skipgram model.
"""

import argparse
import tensorflow as tf
import numpy as np

from relational_sgd.sampling import adapters, factories

from scripts.dataset_logic import load_data_node2vec

from relational_sgd.models.skipgram import make_multilabel_logistic_regression


def parse_arguments(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--train-dir', type=str, default=None)
    parser.add_argument('--max-steps', type=int, default=50000)
    parser.add_argument('--proportion-censored', type=float, default=0.5,
                        help='proportion of censored vertex labels at train time.')
    parser.add_argument('--label-task-weight', type=float, default=1e-3,
                        help='weight to assign to label task.')
    parser.add_argument('--batch-size', type=int, default=None, help='minibatch size')
    parser.add_argument('--dataset-shards', type=int, default=None, help='dataset parallelism')
    parser.add_argument('--use-xla', action='store_true', help='use XLA JIT compilation')
    parser.add_argument('--exotic-evaluation', action='store_true', help='perform exotic evaluation.')

    parser.add_argument('--sampler', type=str, default=None, choices=factories.dataset_names(),
                        help='the sampler to use')

    parser.add_argument('--sampler-test', type=str, default=None, choices=factories.dataset_names(),
                        help='if not None, the sampler to use for testing')

    # defaults set to match Node2Vec
    parser.add_argument('--embedding-dim', type=int, default=128,
                        help='Number of dimensions. Default is 128.')

    parser.add_argument('--num-edges', type=int, default=800,
                        help='Number of edges per sample.')

    parser.add_argument('--window-size', type=int, default=10,
                        help='Context size for optimization. Default is 10.')

    parser.add_argument('--num-negative', type=int, default=5,
                        help='negative examples per vertex for negative sampling')

    parser.add_argument('--num-negative-total', type=int, default=None,
                        help='total number of negative vertices sampled')

    parser.add_argument('--embedding_learning_rate', type=float, default=0.025,
                        help='sgd learning rate for embedding updates')

    parser.add_argument('--global_learning_rate', type=float, default=1.,
                        help='sgd learning rate for global updates')

    parser.add_argument('--global_regularization', type=float, default=1.,
                        help='regularization scale for global variables')

    return parser.parse_args()


def get_dataset_fn(sampler, args):
    if sampler is None:
        sampler = 'biased-walk'

    return factories.make_dataset(sampler, args)


def make_input_fn(graph_data, args, dataset_fn=None, num_samples=None):
    def input_fn():
        dataset = dataset_fn(graph_data, args.seed)

        data_processing = adapters.compose(
            adapters.relabel_subgraph(),
            adapters.append_vertex_labels(graph_data.labels),
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


def _adjust_learning_rate(learning_rate, batch_size):
    if batch_size is not None:
        return learning_rate * batch_size

    return learning_rate


def _adjust_regularization(regularization, batch_size):
    if batch_size is not None:
        return regularization / batch_size

    return regularization


def _make_global_optimizer(args):
    def fn():
        learning_rate = args.global_learning_rate
        return tf.train.GradientDescentOptimizer(
            _adjust_learning_rate(learning_rate, args.batch_size))
    return fn


def make_n2v_test_dataset_fn(args, graph_data):
    rng = np.random.RandomState(args.seed)
    in_train = rng.binomial(1, 1 - args.proportion_censored, size=graph_data.num_vertices).astype(np.int32)

    pred_features = {'vertex_index': np.expand_dims(np.array(range(graph_data.num_vertices)), 1),
                     'is_positive': np.expand_dims(np.array(range(graph_data.num_vertices)), 1)}
    pred_labels = {'labels': np.expand_dims(graph_data.labels,1),
                   'label_index': np.expand_dims(np.array(range(graph_data.num_vertices)),1),
                   'split': np.expand_dims(in_train,1)}

    def n2v_test_dataset_fn():
        dataset = tf.data.Dataset.from_tensor_slices((pred_features, pred_labels))
        return dataset

    return n2v_test_dataset_fn


def main():
    tf.logging.set_verbosity(tf.logging.INFO)
    args = parse_arguments()

    # graph_data = load_data_node2vec()
    graph_data = load_data_node2vec(args.data_dir)

    sg_model = make_multilabel_logistic_regression(
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

    session_config = tf.ConfigProto()

    if args.use_xla:
        session_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

    run_config = tf.estimator.RunConfig(log_step_count_steps=1000, session_config=session_config)

    node_classifier = tf.estimator.Estimator(
        model_fn=sg_model,
        params={
            **vertex_embedding_params,
            'n_labels': graph_data.num_labels,
            'num_vertices': graph_data.num_vertices,
            'batch_size': args.batch_size
        },
        model_dir=args.train_dir,
        config=run_config)

    hooks = [
        tf.train.LoggingTensorHook({
            'kappa_insample': 'kappa_insample_batch/value',
            'kappa_outsample': 'kappa_outsample_batch/value',
            'kappa_edges': 'kappa_edges_in_batch/value'},
            every_n_iter=100)
    ]

    if args.profile:
        hooks.append(tf.train.ProfilerHook(save_secs=30))

    if args.debug:
        from tensorflow.python import debug as tfdbg
        hooks.append(tfdbg.TensorBoardDebugHook('localhost:6004'))

    dataset_fn_train = get_dataset_fn(args.sampler, args)

    node_classifier.train(
        input_fn=make_input_fn(graph_data, args, dataset_fn_train),
        max_steps=args.max_steps,
        hooks=hooks)

    # standard node2vec evaluation
    print("===================")
    print("Node2Vec Evaluation")
    print("===================")

    node_classifier.evaluate(input_fn=make_n2v_test_dataset_fn(args, graph_data),
                             name="node2vec_eval")

    print("===================")
    print("Exotic Evaluation")
    print("===================")

    if args.exotic_evaluation:
        samplers = factories.dataset_names()
        for sampler in samplers:
            dataset_fn_test = get_dataset_fn(sampler, args)

            node_classifier.evaluate(
                input_fn=make_input_fn(graph_data, args, dataset_fn_test, 2000),
                name=sampler)


if __name__ == '__main__':
    main()
