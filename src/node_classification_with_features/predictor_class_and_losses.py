"""
Example defining predictor class and losses for relational ERM model
"""

import tensorflow as tf
from node_classification_with_features.node_classification_template import make_node_classifier
from relational_erm.models.skipgram import _make_edge_list_logits, make_simple_skipgram_loss, make_weighted_loss


def make_nn_class_predictor(label_task_weight=0.001, regularization=0., clip=None, **kwargs):
    """
    Collects a predictor class and loss function to target the problem of classifying nodes using graph structure
    and (real-vector valued) node covariates


    Note: this is a demo chosen for simplicity, and (probably) doesn't actually perform very well

    Parameters
    ----------
    label_task_weight: the weight for the label task (between 0 and 1). Default 0.001
    regularization: regularization applied to neural net
    clip: if not None, the value to clip the edge loss at.
    kwargs: additional arguments are forwarded to the `make_node_classifier` template.

    Returns
    -------
    A model function that combines node embeddings and node features to predict node labels (via a simple neural net)
    """

    # node classifer logits
    def make_label_logits(embeddings, features, mode, params):
        regularizer = tf.contrib.layers.l2_regularizer(scale=label_task_weight * regularization)

        vertex_features = features['vertex_features']
        embedding_and_features = tf.concat([embeddings, vertex_features], axis=-1)

        for units in params['hidden_units']:
            net = tf.layers.dense(embedding_and_features, units=units, activation=tf.nn.relu)

        last_layer = tf.layers.dense(
            net, params['n_classes'], activation=None, use_bias=True,
            kernel_regularizer=regularizer,
            bias_regularizer=regularizer,
            name='logits_labels')

        return last_layer

    edge_task_weight = 1 - label_task_weight

    # node classifier loss
    def make_label_softmax_cross_entropy_loss(logits, classes, split):
        """ Helper function to create label loss

        Parameters
        ----------
        logits: tensor of shape [batch_size, num_verts, num_classes]
        classes: tensor of shape [batch_size, num_verts]; the true classes
        split: tensor of shape [batch_size, num_verts], 0 if censored, 1 if not censored

        Returns
        -------
        The softmax cross-entropy loss of the prediction on the label.
        """
        if len(logits.shape) == 3:
            batch_size = tf.to_float(tf.shape(logits)[0])
        else:
            batch_size = 1

        label_pred_losses = tf.losses.sparse_softmax_cross_entropy(
            classes, logits=logits, weights=split, reduction=tf.losses.Reduction.NONE)

        # sum rather than (tf default of) mean because ¯\_(ツ)_/¯
        label_pred_loss = tf.reduce_sum(label_pred_losses)

        return label_pred_loss / batch_size

    # subgraph prediction and loss are the standard skipgram approach (so we just import them)
    return make_node_classifier(
        make_label_logits=make_label_logits,
        make_edge_logits=_make_edge_list_logits,
        # make_weighted_loss balances class and graph prediction losses
        make_label_pred_loss=make_weighted_loss(make_label_softmax_cross_entropy_loss, label_task_weight),
        make_edge_pred_loss=make_weighted_loss(make_simple_skipgram_loss(clip), edge_task_weight),
        **kwargs)
