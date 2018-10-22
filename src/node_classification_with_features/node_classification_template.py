"""
Example defining tensorflow model function for relational ERM model

This class of models assign an embedding vector to each node and use these embeddings, in conjunction with node
features, to predict graph structure and node classes.
"""

import tensorflow as tf

from relational_erm.models.multilabel_node_classification_template import _get_value, \
    _make_dataset_summaries, _default_embedding_optimizer, _default_global_optimizer, _make_embedding_variable
from relational_erm.models import metrics


def _make_metrics(labels, predictions, weights):
    assert weights is not None

    accuracy = tf.metrics.accuracy(
        labels=labels,
        predictions=predictions,
        weights=weights)

    precision = tf.metrics.precision(
        labels=labels,
        predictions=predictions,
        weights=weights)

    recall = tf.metrics.recall(
        labels=labels,
        predictions=predictions,
        weights=weights)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
    }


def _make_label_prediction_summaries(labels, predicted_labels, split):
    """ Make summaries for label prediction task.

    Parameter
    ---------
    labels:  the labels of the nodes.
    predicted_labels: the predicted labels.
    split: indicates whether the label of each node is censored.
    split == 1 indicates insample, wherease split == 0 indicates out of sample.
    split == -1 denotes fake padded values.
    """
    split_insample = tf.to_float(tf.equal(split, 1))
    split_outsample =tf.to_float(tf.equal(split, 0))

    accuracy_batch_insample = metrics.batch_accuracy(
        labels, predicted_labels, split_insample,
        name='accuracy_insample_batch')

    accuracy_batch_outsample = metrics.batch_accuracy(
        labels, predicted_labels, split_outsample,
        name='accuracy_outsample_batch'
    )

    tf.summary.scalar('accuracy_batch_in', accuracy_batch_insample)
    tf.summary.scalar('accuracy_batch_out', accuracy_batch_outsample)


def make_node_classifier(make_label_logits,
                         make_edge_logits,
                         make_label_pred_loss,
                         make_edge_pred_loss,
                         embedding_optimizer=None,
                         global_optimizer=None):
    """ Creates a node classifier function from various parts.

    Parameters
    ----------
    make_label_logits: function (embeddings, features, mode, params) -> (logits),
        which computes the label logits for for each node.
    make_edge_logits: function (embeddings, features, edge_list, edge_weights, params) -> (label_logits),
        which computes the logits for each pair in edge_list.
    make_label_pred_loss: function (label_logits, vertex_classes) -> (losses),
        which computes the label prediction loss.
    make_edge_pred_loss: function (embeddings, n_vert, el, w, params) -> (losses),
        which computes the edge prediction loss.
    embedding_optimizer: the optimizer (or a nullary function creating the optimizer) to use for the embedding variables.
    global_optimizer: the optimizer (or a nullary function creating the optimizer) to use for the global variables.

    Returns
    -------
    node_classifier: function, to be passed as model_fn to a node classification tensorflow estimator
    """
    if embedding_optimizer is None:
        embedding_optimizer = _default_embedding_optimizer

    if global_optimizer is None:
        global_optimizer = _default_global_optimizer

    def node_classifier(features, labels, mode, params):
        """ The model function for the node classifier.

        Parameters
        ----------
        features: dictionary of graph attributes {edge list, weights, ids of sampled vertices},
            and possibly additional vertex attributes
        labels: dictionary of labels and friends. labels is tensor containing labels of the vertices in the sample
        mode: the estimator mode in which this model function is invoked.
        params: a dictionary of parameters.

        Returns
        -------
        estimator_spec: the estimator spec for the given problem.
        """
        # node embeddings
        all_embeddings = _make_embedding_variable(params)

        # vertices in the sampled subgraph
        vertex_index = features['vertex_index']

        vertex_embedding_shape = tf.concat(
            [tf.shape(vertex_index), [params['embedding_dim']]], axis=0,
            name='vertex_embedding_shape')

        # all embeddings has shape [num_verts, embedding_dim]
        # for compatibility with batching, we flatten the vertex index prior to extracting embeddings
        embeddings = tf.nn.embedding_lookup(all_embeddings, tf.reshape(vertex_index, [-1]))
        embeddings = tf.reshape(embeddings, vertex_embedding_shape, name='vertex_embeddings_batch')

        # Vertex Label Predictions
        vertex_classes = labels['classes']
        split = labels['split']

        with tf.variable_scope("label_logits"):
            label_logits = make_label_logits(embeddings, features, mode, params)

        predicted_classes = tf.argmax(label_logits, -1, output_type=tf.int32)

        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                'class_ids': predicted_classes,
                'probabilities': tf.nn.softmax(label_logits, axis=-1),
                'label_logits': label_logits,
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        """
        label loss and evaluation
        """
        with tf.name_scope('label_loss', values=[label_logits, vertex_classes, split]):
            label_pred_loss = make_label_pred_loss(
                label_logits, vertex_classes,
                tf.maximum(split, 0))  # clip the split, as -1 represents padded values.

        if mode == tf.estimator.ModeKeys.EVAL:
            # Metrics
            estimator_metrics = {}

            with tf.variable_scope('metrics_insample'):
                estimator_metrics.update({
                    k + '_insample': v
                    for k, v in _make_metrics(
                        vertex_classes,
                        predicted_classes,
                        split).items()
                })

            with tf.variable_scope('metrics_outsample'):
                estimator_metrics.update({
                    k + '_outsample': v
                    for k, v in _make_metrics(
                        vertex_classes,
                        predicted_classes,
                        (1 - split)).items()
                })

            return tf.estimator.EstimatorSpec(
                mode, loss=label_pred_loss, eval_metric_ops=estimator_metrics)

        """
        Subgraph structure (edge predictions and loss)
        """
        edge_list = features['edge_list']
        weights = features['weights']
        if weights.shape[-1].value == 1:
            weights = tf.squeeze(weights, axis=-1)

        n_vert = tf.shape(features['vertex_index'])

        # Edge predictions
        edge_logits = make_edge_logits(embeddings, features, edge_list, weights, params)

        # edge loss
        with tf.name_scope('edge_loss', values=[edge_logits, edge_list, weights]):
            edge_pred_loss = make_edge_pred_loss(edge_logits, n_vert, edge_list, weights, params)

            edge_pred_size = tf.shape(edge_logits)[-1]
            edge_pred_loss_normalized = tf.divide(edge_pred_loss, tf.to_float(edge_pred_size))

        reg_loss = tf.losses.get_regularization_loss()

        loss = label_pred_loss + edge_pred_loss + reg_loss

        tf.summary.scalar('label_loss', label_pred_loss, family='loss')
        tf.summary.scalar('edge_loss', edge_pred_loss, family='loss')
        tf.summary.scalar('edge_loss_normalized', edge_pred_loss_normalized, family='loss')
        tf.summary.scalar('regularization_loss', reg_loss, family='loss')

        """
        Summaries 
        """
        _make_label_prediction_summaries(vertex_classes, predicted_classes, split)

        # edge prediction summaries
        predicted_edges = tf.cast(tf.greater(edge_logits, 0.), edge_logits.dtype)
        kappa_batch_edges = metrics.batch_kappa(
            weights, predicted_edges,
            tf.to_float(tf.not_equal(weights, -1)),  # -1 weight indicates padded edges
            name='kappa_edges_in_batch'
        )

        tf.summary.scalar('kappa_batch_edges', kappa_batch_edges)

        # dataset summaries
        _make_dataset_summaries(features, mode)

        """
        Training
        """
        if mode == tf.estimator.ModeKeys.TRAIN:
            batch_size = params['batch_size'] if params['batch_size'] is not None else 1

            embedding_vars = [v for v in tf.trainable_variables() if "embedding" in v.name]
            global_vars = [v for v in tf.trainable_variables() if "embedding" not in v.name]
            global_step = tf.train.get_or_create_global_step()

            update_global_step = tf.assign_add(global_step, batch_size, name="global_step_update")

            embedding_optimizer_value = _get_value(embedding_optimizer)
            global_optimizer_value = _get_value(global_optimizer)

            if len(embedding_vars) > 0:
                embedding_update = embedding_optimizer_value.minimize(
                    loss, var_list=embedding_vars, global_step=None)
            else:
                embedding_update = tf.identity(0.)  # meaningless

            if len(global_vars) > 0:
                global_update = global_optimizer_value.minimize(
                    loss, var_list=global_vars, global_step=None)
            else:
                global_update = tf.identity(0.)

            with tf.control_dependencies([update_global_step]):
                train_op = tf.group(embedding_update, global_update)

            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    return node_classifier
