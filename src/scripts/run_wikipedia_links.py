import tensorflow as tf
import numpy as np
import collections

from tensorflow.contrib.training import HParams

import relational_erm.sampling.negative_sampling
from relational_erm.tensorflow_ops.dataset_ops import RandomWalkDataset
from relational_erm.sampling import adapters
from relational_erm.models import label_clustering

from .dataset_logic import load_data_wikipedia_hyperlink


def _constant_hidden_value(value, name):
    return tf.py_func(
        lambda: value,
        [], tf.int32, stateful=False,
        name=name)


def make_input_fn(adjacency_list, packed_labels, dataset_shards=None):
    num_vertex = len(adjacency_list.lengths)

    def input_fn(params):
        def _dataset_fn(s):
            walk_length = int(params.num_edges / params.window_size)

            return RandomWalkDataset(
                walk_length,
                _constant_hidden_value(adjacency_list.neighbours, 'create_neighbours'),
                _constant_hidden_value(adjacency_list.lengths, 'create_lengths'),
                _constant_hidden_value(adjacency_list.offsets, 'create_offsets'),
                seed=s).prefetch(params.batch_size * 2)

        if dataset_shards is None:
            dataset = _dataset_fn(params.seed)
        else:
            from tensorflow.contrib.data import parallel_interleave

            dataset = tf.data.Dataset.range(dataset_shards).apply(
                parallel_interleave(_dataset_fn, cycle_length=dataset_shards,
                                    sloppy=True))

        window = adapters.adapt_random_walk_window(params.window_size)
        add_negative_samples = relational_erm.sampling.negative_sampling.add_negative_sample(
            num_vertex, params.num_negative, seed=params.seed)

        processing_fn = adapters.compose(
            window,
            add_negative_samples,
            adapters.relabel_subgraph(),
            adapters.append_packed_vertex_labels(
                packed_labels.labels, packed_labels.lengths, packed_labels.offsets),
            adapters.add_sample_size_info(),
            adapters.format_features_labels())

        dataset = dataset.map(processing_fn, num_parallel_calls=12)
        dataset = dataset.prefetch(params.batch_size * 2)
        dataset = dataset.apply(adapters.padded_batch_samples(params.batch_size))
        return dataset.apply(tf.contrib.data.prefetch_to_device('/gpu:0'))

    return input_fn


PackedLabels = collections.namedtuple('PackedLabels', ['labels', 'lengths', 'offsets'])


def get_packed_labels(sparse_labels):
    idx = np.lexsort(sparse_labels.T)
    _, lengths = np.unique(sparse_labels[:, 1], return_counts=True)
    lengths = lengths.astype(np.int32, copy=False)

    packed_labels = sparse_labels[idx, 0]
    offsets = np.empty_like(lengths)
    offsets[0] = 0
    np.cumsum(lengths[:-1], out=offsets[1:])
    return PackedLabels(packed_labels, lengths, offsets)


def main():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--train-dir', type=str, default=None)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--hparams', type=str, default=None)
    parser.add_argument('--max-steps', type=int, default=100000)
    parser.add_argument('--dataset-shards', type=int, default=None)

    args = parser.parse_args()

    tf.logging.set_verbosity(tf.logging.INFO)

    hparams = HParams(num_edges=800,
                      window_size=10,
                      num_negative=5,
                      batch_size=args.batch_size,
                      embedding_dim=128,
                      seed=42)

    if args.hparams is not None:
        hparams.parse(args.hparams)

    wikipedia_data = load_data_wikipedia_hyperlink()
    packed_labels = get_packed_labels(wikipedia_data['labels_sparse'])
    num_labels = np.max(packed_labels.labels) + 1

    input_fn = make_input_fn(wikipedia_data['adjacency_list'],
                             packed_labels,
                             args.dataset_shards)

    estimator = tf.estimator.Estimator(
        label_clustering.make_label_clustering(num_labels),
        model_dir=args.train_dir, params=hparams,
        config=tf.estimator.RunConfig(
            tf_random_seed=hparams.seed))

    hooks = [
        tf.train.LoggingTensorHook({
            'kappa_edges': 'kappa_edges_in_batch/value'},
            every_n_secs=30)]

    if args.profile:
        hooks.append(tf.train.ProfilerHook(save_secs=10))

    estimator.train(input_fn, max_steps=args.max_steps, hooks=hooks)


if __name__ == '__main__':
    main()
