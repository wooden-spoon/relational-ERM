import os
import numpy as np
import tensorflow as tf

import relational_erm.tensorflow_ops.array_ops as array_ops

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def test_concatenate_slices():
    data = np.random.randn(100)
    starts = np.random.randint(90, size=10)
    lengths = np.random.randint(10, size=10)

    concat_slices = array_ops.concatenate_slices(data, starts, lengths)
    concat_slices_numpy = np.concatenate([data[s:s+l] for s, l in zip(starts, lengths)])

    assert np.all(concat_slices == concat_slices_numpy)


def test_packed_to_sparse_index():
    lengths = np.array([3, 1, 2])

    packed_to_sparse = array_ops.packed_to_sparse_index(lengths)

    assert packed_to_sparse.shape == (lengths.sum(), 2)
    assert np.all(packed_to_sparse == [[0, 0], [0, 1], [0, 2], [1, 0], [2, 0], [2, 1]])


def test_repeat_op():
    values = np.random.randn(5)
    counts = np.array([2, 3, 5, 7, 3])

    repeated = array_ops.repeat(values, counts)

    assert np.all(repeated == np.repeat(values, counts))


def test_batch_segment_op():
    lengths = [[2, 3], [2, 0]]
    output_columns = 5

    batch_segment = array_ops.batch_length_to_segment(lengths, output_columns)

    assert np.all(batch_segment == [[0, 0, 1, 1, 1], [3, 3, 5, 5, 5]])


def test_batch_segment_op_padding():
    lengths = [[1, 2, 0], [1, 0, 0]]
    output_columns = 3

    batch_segment = array_ops.batch_length_to_segment(lengths, output_columns)

    assert np.all(batch_segment == [[0, 1, 1], [4, 7, 7]])
