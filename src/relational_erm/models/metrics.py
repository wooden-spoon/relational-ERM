""" Implementation of metrics. """

import tensorflow as tf


def oracle_predictions(labels, logits, name=None):
    """ Compute the predictions from the given logits where the threshold is
    chosen in an oracle fashion based on the true labels.

    Parameters
    ----------
    labels: a tensor of shape [num_batch, num_labels] taking values in {0, 1}.
    logits: a tensor of the same shape as labels representing the unnormalized log-probabilities
        of a label being present.
    name: the name of the operation.

    Returns
    -------
    predictions: the predictions obtained from the oracle threshold.
    """
    with tf.name_scope(name, 'oracle_predictions', values=[labels, logits]):
        num_true = tf.to_int32(tf.reduce_sum(labels, axis=-1, name='num_true', keepdims=True))
        num_true_flat = tf.reshape(num_true, [-1])

        logits_sorted, _ = tf.nn.top_k(logits, logits.shape[-1], sorted=True)
        logits_sorted_flat = tf.reshape(logits_sorted, [-1, logits.shape[-1]])

        nth_logit_index = tf.stack(
            [tf.range(tf.size(num_true_flat), dtype=num_true_flat.dtype), num_true_flat],
            axis=1)

        logits_values_flat = tf.gather_nd(logits_sorted_flat, nth_logit_index, name='top_k_value_flat')
        logits_values = tf.reshape(logits_values_flat, tf.shape(logits)[:-1])

        predictions = tf.greater_equal(logits, tf.expand_dims(logits_values, -1))
        predictions = tf.to_float(predictions)

    return predictions


def macro_f1(predictions, labels, weights=None,
             metrics_collections=None,
             updates_collections=None,
             name=None):
    """ Computes the averaged f1 score across a number of prediction tasks.

    Parameters
    ----------
    predictions: a tensor of dimension at least 2 representing the predictions
        for each task. The last dimension represents the different tasks.
    labels: a tensor of the same shape as predictions representing the labels.
    weights: a tensor of dimension one less than predictions, representing
        the weight of each observation.
    metrics_collections: a list of collections into which to add the metric values.
    updates_collections: a list of collections into which to add the update ops.
    name: an optional name to give to the operation

    Returns
    -------
    f1_value: the macro f1 value
    update: the operation to update the metric.
    """
    metric_variable_collections = [tf.GraphKeys.LOCAL_VARIABLES]

    with tf.variable_scope(name, default_name='macro_f1', values=[predictions, labels, weights]):
        num_classes = predictions.shape[-1]

        true_positives = tf.get_variable('true_positives', shape=num_classes, dtype=tf.int64,
                                         initializer=tf.zeros_initializer,
                                         trainable=False,
                                         collections=metric_variable_collections)

        false_positives = tf.get_variable('false_positive', shape=num_classes, dtype=tf.int64,
                                          initializer=tf.zeros_initializer,
                                          trainable=False,
                                          collections=metric_variable_collections)

        false_negatives = tf.get_variable('true_negatives', shape=num_classes, dtype=tf.int64,
                                          initializer=tf.zeros_initializer,
                                          trainable=False,
                                          collections=metric_variable_collections)

        with tf.name_scope('update'):
            if weights is not None:
                weight_broadcast = tf.expand_dims(weights, -1)
            else:
                weight_broadcast = 1

            reduction_axes = list(range(len(predictions.shape) - 1))
            weight_broadcast = tf.cast(weight_broadcast, tf.int64)

            predictions = tf.not_equal(predictions, 0)
            labels = tf.not_equal(labels, 0)

            assign_true_pos = tf.assign_add(
                true_positives,
                tf.reduce_sum(
                    tf.to_int64(tf.logical_and(predictions, labels)) * weight_broadcast,
                    axis=reduction_axes),
                name='assign_true_positives')
            assign_false_pos = tf.assign_add(
                false_positives,
                tf.reduce_sum(
                    tf.to_int64(tf.logical_and(predictions, tf.logical_not(labels))) * weight_broadcast,
                    axis=reduction_axes),
                name='assign_true_negatives')
            assign_false_neg = tf.assign_add(
                false_negatives,
                tf.reduce_sum(
                    tf.to_int64(tf.logical_and(tf.logical_not(predictions), labels)) * weight_broadcast,
                    axis=reduction_axes),
                name='assign_false_negatives')

            update = tf.group(assign_true_pos, assign_false_pos, assign_false_neg, name='compute_summary')

        with tf.name_scope('value'):
            precision = tf.where(
                tf.greater(true_positives + false_positives, 0),
                tf.divide(true_positives, true_positives + false_positives),
                tf.zeros_like(true_positives, dtype=tf.float64),
                name='precision')

            recall = tf.where(
                tf.greater(true_positives + false_negatives, 0),
                tf.divide(true_positives, true_positives + false_negatives, name='recall'),
                tf.zeros_like(true_positives, dtype=tf.float64),
                name='recall')

            f1_per_label = 2 / (1 / recall + 1 / precision)
            f1_value = tf.reduce_mean(f1_per_label, name='macro_f1')

    if updates_collections is not None:
        for collection in updates_collections:
            tf.add_to_collection(collection, update)

    if metrics_collections is not None:
        for collection in metrics_collections:
            tf.add_to_collection(collection, f1_value)

    return f1_value, update


def macro_f1_oracle(logits, labels, weights=None,
                    metrics_collections=None,
                    updates_collections=None,
                    name=None):
    """ Computes the averaged f1 score across a number of prediction tasks, where predictions
    are obtained from the logits in an oracle fashion.

    Parameters
    ----------
    logits: a tensor of dimension at least 2 representing the unnormalized probabilities.
    labels: a tensor of the same shape as predictions representing the labels.
    weights: a tensor of dimension one less than predictions, representing
        the weight of each observation.
    metrics_collections: a list of collections into which to add the metric values.
    updates_collections: a list of collections into which to add the update ops.
    name: an optional name to give to the operation

    Returns
    -------
    f1_value: the macro f1 value
    update: the operation to update the metric.
    """
    with tf.name_scope(name, "macro_f1_oracle", values=[logits, labels, weights]):
        predictions = oracle_predictions(labels, logits)

        return macro_f1(predictions, labels, weights, metrics_collections, updates_collections)


def batch_accuracy(labels, predictions, weights, name=None):
    """ Computes the accuracy on the given batch of predictions

    labels: a tensor of any shape.
    predictions: a tensor of the same shape as labels.
    weights: a tensor that may be broadcasted to the shape of labels.
    name: an optional name for the operation

    Returns
    -------
    accuracy_batch: a scalar tensor representing the batch accuracy.
    """
    with tf.name_scope(name, 'batch_accuracy', [labels, predictions, weights]):
        weights_mean = tf.reduce_mean(weights)

        accuracy_batch = tf.reduce_mean(
            tf.divide(
                tf.to_float(tf.equal(labels, predictions)) * weights,
                tf.where(tf.not_equal(weights_mean, 0), weights_mean, 1)),
            name='value')

    return accuracy_batch


def batch_random_agreement(labels, predictions, weights, name=None):
    """ Computes the probability of random agreement between the
    labels and predictions assuming independence.

    Parameters
    ----------
    labels: a tensor of any shape taking values in {0, 1}.
    predictions: a tensor of the same shape as labels taking values in {0, 1}.
    weights: a tensor that can be broadcasted to labels.
    name: an optional name for the operation.

    Returns
    -------
    random_agreement: a scalar tensor representing the probability of random
        agreement.
    """
    with tf.name_scope(name, 'batch_random_agreement', [labels, predictions, weights]):
        weights_mean = tf.reduce_mean(weights)
        weights_mean = tf.where(tf.not_equal(weights_mean, 0), weights_mean, 1)

        p_labels = tf.reduce_mean(labels * weights) / weights_mean
        p_predictions = tf.reduce_mean(predictions * weights) / weights_mean

        random_agreement = tf.identity(
            p_labels * p_predictions + (1 - p_labels) * (1 - p_predictions),
            name='value')

    return random_agreement


def batch_kappa(labels, predictions, weights, name=None):
    """ Computes Cohen's kappa on the given batch of predictions.

    Parameters
    ----------
    labels: a tensor of any shape taking values in {0, 1}.
    predictions: a tensor of the same shape as labels taking values in {0, 1}.
    weights: a tensor that can be broadcasted to labels.
    name: an optional name for the operation.

    Returns
    -------
    kappa: a scalar tensor representing the Kappa measure of agreement
        between labels and predictions.
    """
    with tf.name_scope(name, 'batch_kappa', [labels, predictions, weights]):
        accuracy = batch_accuracy(labels, predictions, weights)
        random_agreement = batch_random_agreement(labels, predictions, weights)

        kappa = tf.divide(
            accuracy - random_agreement, 1 - random_agreement,
            name='value')

    return kappa


def batch_kappa_oracle(labels, logits, weights, name=None):
    """ Computes Cohen's Kappa on the given batch with the given logits,
    where the predictions are generated given the oracle number of true labels
    for each sample.

    Parameters
    ----------
    labels: a tensor of shape [num_batch, num_labels] taking values in {0, 1}.
    logits: a tensor of the same shape as labels representing the unnormalized log-probabilities
        of a label being present.
    weights: a tensor that can be broadcasted to labels.
    name: an optional name for the operation

    Returns
    -------
    kappa: a scalar tensor representing the Kappa measure of agreement.
    """
    with tf.name_scope(name, 'batch_kappa_oracle', [labels, logits, weights]):
        predictions = oracle_predictions(labels, logits)

        return batch_kappa(labels, predictions, weights)
