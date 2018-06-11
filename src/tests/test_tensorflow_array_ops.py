import os
import numpy as np
import tensorflow as tf

import relational_sgd.tensorflow_ops.array_ops as array_ops

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def test_concatenate_slices():
    data = np.random.randn(100)
    starts = np.random.randint(90, size=10)
    lengths = np.random.randint(10, size=10)

    with tf.Graph().as_default():
        concat_slices_op = array_ops.concatenate_slices(data, starts, lengths)

        with tf.Session() as session:
            concat_slices = session.run(concat_slices_op)

    concat_slices_numpy = np.concatenate([data[s:s+l] for s, l in zip(starts, lengths)])

    assert np.all(concat_slices == concat_slices_numpy)


def test_packed_to_sparse_index():
    lengths = np.array([3, 1, 2])

    with tf.Graph().as_default():
        packed_to_sparse_op = array_ops.packed_to_sparse_index(lengths)

        with tf.Session() as session:
            packed_to_sparse = session.run(packed_to_sparse_op)

    assert packed_to_sparse.shape == (lengths.sum(), 2)
    assert np.all(packed_to_sparse == [[0, 0], [0, 1], [0, 2], [1, 0], [2, 0], [2, 1]])


def test_repeat_op():
    values = np.random.randn(5)
    counts = np.array([2, 3, 5, 7, 3])

    with tf.Graph().as_default():
        repeat_op = array_ops.repeat(values, counts)

        with tf.Session() as session:
            repeated = session.run(repeat_op)

    assert np.all(repeated == np.repeat(values, counts))


def test_batch_segment_op():
    lengths = [[2, 3], [2, 0]]
    output_columns = 5

    with tf.Graph().as_default():
        batch_segment_op = array_ops.batch_length_to_segment(lengths, output_columns)

        with tf.Session() as session:
            batch_segment = session.run(batch_segment_op)

    assert np.all(batch_segment == [[0, 0, 1, 1, 1], [3, 3, 5, 5, 5]])


def test_batch_segment_op_padding():
    lengths = [[1, 2, 0], [1, 0, 0]]
    output_columns = 3

    with tf.Graph().as_default():
        batch_segment_op = array_ops.batch_length_to_segment(lengths, output_columns)

        with tf.Session() as session:
            batch_segment = session.run(batch_segment_op)

    assert np.all(batch_segment == [[0, 1, 1], [4, 7, 7]])
