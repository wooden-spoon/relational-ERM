""" Model for label clustering

"""

import tensorflow as tf

from relational_sgd.tensorflow_ops import array_ops
from . import metrics


def _make_label_embeddings(num_labels, params):
    embedding_variable_name = 'label_embeddings'

    embeddings = tf.get_variable(
        embedding_variable_name,
        shape=[num_labels, params.embedding_dim],
        dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=1 / params.embedding_dim),
        trainable=True)

    return embeddings


def _load_vertex_embeddings(labels, embeddings):
    """ Creates a sparse tensor which describes the labels attached to each vertex.

    Parameters
    ----------
    labels: dictionary of label data.
    """
    packed_labels = labels['packed_labels']
    packed_labels_lengths = labels['packed_labels_lengths']

    packed_labels_shape = tf.shape(packed_labels)
    packed_labels_lengths_shape = tf.shape(packed_labels_lengths)
    packed_labels_flat = tf.reshape(packed_labels, [-1])

    label_embeddings_flat = tf.nn.embedding_lookup(embeddings, packed_labels_flat)

    if len(packed_labels_lengths.shape) > 1:
        segments_flat = tf.reshape(
            array_ops.batch_length_to_segment(packed_labels_lengths, packed_labels_shape[-1]),
            [-1])
    else:
        segments_flat = array_ops.repeat(
            tf.range(tf.size(packed_labels_lengths), dtype=tf.int32),
            packed_labels_lengths)

    embeddings_flat = tf.segment_sum(
        label_embeddings_flat, segments_flat)

    output_embedding_shape = tf.stack(
        (packed_labels_lengths_shape[0], packed_labels_lengths_shape[1] + 1, embeddings.shape[1]),
        axis=0, name='vertex_embedding_shape')

    if len(packed_labels_lengths.shape) > 1:
        expected_num_segments = packed_labels_lengths_shape[0] * (packed_labels_lengths_shape[1] + 1)
        length_to_pad = expected_num_segments - tf.shape(embeddings_flat)[0]

        embeddings_flat = tf.pad(
            embeddings_flat,
            tf.reshape(tf.stack((0, length_to_pad, 0, 0), axis=0), [2, 2]))

    embeddings = tf.reshape(
        embeddings_flat,
        output_embedding_shape)

    return embeddings


def compute_edge_logits(embeddings, edge_list):
    """ Helper function to create the skipgram loss for edge structure

    Parameters
    ----------
    embeddings: the embeddings features for the current subgraph.
    edge_list: edge list of the subgraph

    Returns
    -------
    a tensor representing the edge prediction loss.
    """
    with tf.name_scope('edge_list_logits'):
        pairwise_inner_prods = tf.matmul(embeddings, embeddings, transpose_b=True,
                                         name='all_edges_logit')

        if len(edge_list.shape) == 2:
            edge_list = tf.expand_dims(edge_list, axis=0)
            pairwise_inner_prods = tf.expand_dims(pairwise_inner_prods, axis=0)
            no_batch = True
        else:
            no_batch = False

        edge_list_shape = tf.shape(edge_list)
        batch_size = edge_list.shape[0].value if edge_list.shape[0].value is not None else edge_list_shape[0]
        num_edges = edge_list.shape[1].value if edge_list.shape[1].value is not None else edge_list_shape[1]

        batch_index = tf.tile(
            tf.expand_dims(tf.expand_dims(tf.range(batch_size), -1), -1),
            tf.stack([1, num_edges, 1]))

        edge_index = tf.concat([batch_index, edge_list], axis=-1)
        edge_logit = tf.gather_nd(pairwise_inner_prods, edge_index)

        if no_batch:
            edge_logit = tf.squeeze(edge_logit, axis=0)

        return edge_logit


def skipgram_loss(edge_logits, edge_list, edge_weights, params):
    with tf.name_scope('skipgram_loss', values=[edge_logits, edge_list, edge_weights]):
        if len(edge_list.shape) == 3:
            batch_size = tf.to_float(tf.shape(edge_list)[0])
        else:
            batch_size = 1.

        edge_present = tf.maximum(edge_weights, 0)

        # values of -1 in the weights indicate padded edges which should be ignored
        # in loss computation.
        edge_censored = tf.to_float(tf.not_equal(edge_weights, -1))

        edge_pred_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=edge_present, logits=edge_logits)

        edge_pred_loss = edge_pred_loss * edge_censored

        if hasattr(params, 'clip') and params.clip:
            edge_pred_loss = tf.clip_by_value(edge_pred_loss, 0, params.clip)

        # sum instead of (tf default of) mean because mean screws up learning rates for embeddings
        loss_value = tf.divide(tf.reduce_sum(edge_pred_loss), batch_size,
                               name='skipgram_edge_loss')
    return loss_value


def make_label_clustering(num_labels):
    def estimator_fn(features, labels, mode, params):
        all_label_embeddings = _make_label_embeddings(num_labels, params)

        vertex_embeddings = _load_vertex_embeddings(labels, all_label_embeddings)

        edge_weights = features['weights']

        if edge_weights.shape[-1] == 1:
            edge_weights = tf.squeeze(edge_weights, axis=-1)

        edge_logits = compute_edge_logits(vertex_embeddings, features['edge_list'])
        edge_loss = skipgram_loss(edge_logits, features['edge_list'], edge_weights, params)

        optimizer = tf.train.GradientDescentOptimizer(0.01)

        train_op = optimizer.minimize(
            edge_loss, tf.train.get_or_create_global_step(), var_list=tf.trainable_variables())

        predicted_edges = tf.cast(tf.greater(edge_logits, 0.), edge_logits.dtype)
        kappa_batch_edges = metrics.batch_kappa(
            edge_weights, predicted_edges,
            tf.to_float(tf.not_equal(edge_weights, -1)),  # -1 weight indicates padded edges
            name='kappa_edges_in_batch')

        tf.summary.scalar('kappa_batch_edges', kappa_batch_edges)

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=edge_loss,
            train_op=train_op
        )

    return estimator_fn
