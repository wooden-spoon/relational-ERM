"""
Example script for training and evaluating relational ERM model
"""

import sys

import tensorflow as tf

from node_classification_with_features.dataset_logic.dataset_logic import load_data_graphsage

from node_classification_with_features.sample import make_sample, augment_sample
from node_classification_with_features.predictor_class_and_losses import make_nn_class_predictor

# helpers for batching
from scripts.run_skipgram_simple import parse_arguments, \
    _adjust_regularization, _adjust_learning_rate, _make_global_optimizer


def main():
    # tf.enable_eager_execution()
    #
    # # create fake args for debugging
    # sys.argv = ['']
    # args = parse_arguments()
    # args.batch_size = 10
    # args.max_steps = 5000

    args = parse_arguments()

    graph_data = load_data_graphsage(args.data_dir)


    """
    Sample defined as a custom tf dataset
    """
    sample_train = make_sample(args.sampler, args)  # sample subgraph according to graph sampling scheme args.sampler
    input_fn = augment_sample(graph_data, args, sample_train)  # augment subgraph with vertex labels and features

    """
    Predictor class and loss function
    """
    # hyperparams
    vertex_embedding_params = {
        'embedding_dim': args.embedding_dim,
        'embedding_trainable': True,
        'embedding_checkpoint': None
    }

    params={
        **vertex_embedding_params,
        'hidden_units' : [200, 200], # Jaan net
        'n_classes':  max(graph_data.classes)+1,
        'num_vertices': graph_data.num_vertices,
        'batch_size': args.batch_size
    }

    classifier_predictor_and_loss = make_nn_class_predictor(
        label_task_weight=args.label_task_weight,
        regularization=_adjust_regularization(args.global_regularization, args.batch_size),
        global_optimizer=_make_global_optimizer(args),
        embedding_optimizer=lambda: tf.compat.v1.train.GradientDescentOptimizer(
            _adjust_learning_rate(args.embedding_learning_rate, args.batch_size))
    )

    node_classifier = tf.estimator.Estimator(
        model_fn=classifier_predictor_and_loss,
        params=params,
        model_dir=args.train_dir)


    """
    Put it together for the optimization
    """
    # some extra logging
    hooks = [
        tf.estimator.LoggingTensorHook({
            'kappa_edges': 'kappa_edges_in_batch/value'},
            every_n_iter=100)
    ]

    if args.profile:
        hooks.append(tf.estimator.ProfilerHook(save_secs=30))

    if args.debug:
        from tensorflow.python import debug as tfdbg
        hooks.append(tfdbg.TensorBoardDebugHook('localhost:6004'))

    node_classifier.train(
        input_fn=input_fn,
        max_steps=args.max_steps,
        hooks=hooks)


    """
    Evaluate
    """
    node_classifier.evaluate(input_fn=augment_sample(graph_data, args, sample_train, 2000),
                             name="node2vec_eval")


if __name__ == '__main__':
    main()
