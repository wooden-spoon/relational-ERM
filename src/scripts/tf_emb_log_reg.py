"""
Tensorflow model using logistic regression to predict vertex labels from pretrained embeddings
"""

import argparse
import os

import tensorflow as tf
import numpy as np

from relational_sgd.sampling import adapters, factories

from scripts.dataset_logic import load_data_node2vec

from relational_sgd.models.skipgram import make_multilabel_logistic_regression
import scripts.run_skipgram_simple
from scripts.run_skipgram_simple import get_dataset_fn, make_input_fn, _adjust_learning_rate, _adjust_regularization

from models.scoring import predict
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-steps-logistic', type=int, default=40000)
    parser.add_argument('--embedding-dir', type=str, default=None)

    return scripts.run_skipgram_simple.parse_arguments(parser)


def make_train_logistic_estimator(embedding_ckpt):
    def train_logistic_estimator(vertex_id, args):
        tensor_name_in_ckpt = "input_layer/vertex_index_embedding/embedding_weights"
        ckpt_file = tf.train.latest_checkpoint(embedding_ckpt)

        vertex_embedding = tf.feature_column.embedding_column(
            vertex_id, dimension=args.embedding_dim,
            tensor_name_in_ckpt=tensor_name_in_ckpt,
            ckpt_to_load_from=ckpt_file,
            trainable=False)

        model = make_multilabel_logistic_regression(
            label_task_weight=1.0,
            regularization=_adjust_regularization(args.global_regularization, args.batch_size),
            global_optimizer=tf.train.GradientDescentOptimizer(
                _adjust_learning_rate(args.global_learning_rate,
                                      args.batch_size)))
        hooks = [tf.train.LoggingTensorHook(
            {'kappa_insample': 'kappa_insample_batch/value',
             'kappa_outsample': 'kappa_outsample_batch/value'},
            every_n_secs=30)]

        return model, vertex_embedding, hooks, None

    return train_logistic_estimator


def make_optimizer(args):
    def fn():
        return tf.train.GradientDescentOptimizer(
            _adjust_learning_rate(args.learning_rate, args.batch_size))

    return fn


def main():
    tf.logging.set_verbosity(tf.logging.INFO)
    args = parse_arguments()

    # graph_data = load_data_node2vec()
    graph_data = load_data_node2vec(args.data_dir)

    vertex_embedding_params = {
        'embedding_dim': args.embedding_dim,
        'embedding_trainable': False,
        'embedding_checkpoint': tf.train.latest_checkpoint(args.embedding_dir),
    }

    model = make_multilabel_logistic_regression(
        label_task_weight=1.0,
        regularization=args.global_regularization,
        global_optimizer=make_optimizer(args),
        polyak=False)
    hooks = [tf.train.LoggingTensorHook(
        {'kappa_insample': 'kappa_insample_batch/value',
         'kappa_outsample': 'kappa_outsample_batch/value'},
        every_n_secs=30)]

    node_classifier = tf.estimator.Estimator(
        model_fn=model,
        params={
            **vertex_embedding_params,
            'num_vertices': graph_data.num_vertices,
            'n_labels': graph_data.num_labels,
            'batch_size': args.batch_size
        },
        model_dir=args.train_dir)

    if args.profile:
        hooks.append(tf.train.ProfilerHook(save_secs=30))

    if args.debug:
        from tensorflow.python import debug as tfdbg
        hooks.append(tfdbg.TensorBoardDebugHook('localhost:6004'))

    # train model
    dataset_fn_train = get_dataset_fn(args.sampler, args)

    node_classifier.train(
        input_fn=make_input_fn(graph_data, args, dataset_fn_train),
        max_steps=args.max_steps_logistic,
        hooks=hooks)

    pred_features = {'vertex_index': np.expand_dims(np.array(range(graph_data.num_vertices)), 1)}

    def make_pred_dataset():
        dataset = tf.data.Dataset.from_tensor_slices(pred_features)
        return dataset

    print('======= Computing Predictions for logistic regression ========')
    predictions = node_classifier.predict(
            input_fn=make_pred_dataset,
            yield_single_examples=False)

    # get test set
    rng = np.random.RandomState(args.seed)
    in_train = rng.binomial(1, 1 - args.proportion_censored, size=graph_data.num_vertices).astype(np.int32)
    in_test = np.logical_not(in_train)

    pred_prob_list = []
    for prediction in predictions:
        pred_prob_list += [prediction['probabilities']]
    pred_probs = np.concatenate(pred_prob_list)

    num_labels = graph_data.labels.shape[1]
    classes = np.array(range(num_labels))

    top_k_list = list(np.sum(graph_data.labels[in_test], 1).astype(np.int))
    pred_labels = predict(pred_probs[in_test], classes, top_k_list)

    mlb = MultiLabelBinarizer(classes)
    pred_labels = mlb.fit_transform(pred_labels)

    print('======= Result for logistic regression ========')
    f1_macro = f1_score(graph_data.labels[in_test], pred_labels, average='macro')
    f1_micro = f1_score(graph_data.labels[in_test], pred_labels, average='micro')
    print("f1_macro: {}".format(f1_macro))
    print("f1_micro: {}".format(f1_micro))

    # test model
    dataset_fn_test = get_dataset_fn(args.sampler_test if args.sampler_test is not None else args.sampler, args)

    node_classifier.evaluate(
        input_fn=make_input_fn(graph_data, args, dataset_fn_test, 1000),
        hooks=hooks)


if __name__ == '__main__':
    main()