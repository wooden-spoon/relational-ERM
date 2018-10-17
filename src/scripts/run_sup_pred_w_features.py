"""
This runs a simple model for semi-supervised vertex label prediction using
graph structure and vertex information
"""

import argparse
import tensorflow as tf
import numpy as np

from relational_sgd.sampling import adapters, factories

from scripts.dataset_logic import load_data_node2vec

from relational_sgd.models.skipgram import make_multilabel_logistic_regression

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
            adapters.append_vertex_labels(graph_data.labels),
            # adapters.split_vertex_labels(
            #     graph_data.num_vertices, args.proportion_censored,
            #     np.random.RandomState(args.seed)),
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


def make_n2v_test_dataset_fn(args, graph_data):
    rng = np.random.RandomState(args.seed)
    in_train = rng.binomial(1, 1 - args.proportion_censored, size=graph_data.num_vertices).astype(np.int32)

    pred_features = {'vertex_index': np.expand_dims(np.array(range(graph_data.num_vertices)), 1),
                     'is_positive': np.expand_dims(np.array(range(graph_data.num_vertices)), 1)}
    pred_labels = {'labels': np.expand_dims(graph_data.labels,1),
                   'split': np.expand_dims(in_train,1)}

    def n2v_test_dataset_fn():
        dataset = tf.data.Dataset.from_tensor_slices((pred_features, pred_labels))
        return dataset

    return n2v_test_dataset_fn


def main():
    graph_data = load_data_node2vec()

    args = namedtuple('dummyArgs', ['num_edges',
                                       'window_size',
                                       'dataset_shards',
                                       'num_negative',
                                       'num_negative_total',
                                       'seed'])

    args.num_edges = 800
    args.window_size = 10
    args.dataset_shards = 1
    args.num_negative = 5
    args.num_negative_total = None
    args.seed = 0

    # testing dataset
    dataset_fn = factories.make_biased_random_walk_dataset(args)
    # dataset = dataset_fn(graph_data, args.seed)
    input_fn = make_input_fn(graph_data, args, dataset_fn)

    ds = input_fn()
    ds_itr = ds.make_initializable_iterator()

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    sess.run(ds_itr.initializer)
    samp = ds_itr.get_next()
    sess.run(samp)

    node_classifier.train(
        input_fn=make_input_fn(graph_data, args, dataset_fn_train),
        max_steps=args.max_steps,
        hooks=hooks)


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
