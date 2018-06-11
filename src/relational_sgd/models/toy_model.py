import tensorflow as tf

def my_model(features, labels, mode, params):
    """

    :param features: dictionary of edge list, weights, ids of sampled vertices, and possibly additional vertex attributes
    :param labels: {verts: [int], labels: [int, int]} where verts is indices of labelled vertices in the subgraph, and labels are labels
    :param mode:
    :param params:
    :return:
    """

    """
    Handles
    """

    embeddings = tf.feature_column.input_layer(features, params['embeddings'])

    el = features['el']
    w = features['w']

    logits = tf.layers.dense(embeddings, params['n_labels'], activation=None)

    """
    Vertex Label Predictions
    """
    # Compute predictions.
    predicted_classes = tf.cast(tf.greater(logits, 0.), logits.dtype)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.sigmoid(logits),
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    """
    Scoring
    """
    # vertex labels
    labelled_verts = labels['verts']
    present_labels = labels['labels']

    labelled_emb = tf.gather(logits, labelled_verts)
    label_pred_loss = tf.losses.sigmoid_cross_entropy(present_labels, logits=labelled_emb)

    # relational structure
    all_edge_pred = tf.matmul(embeddings, embeddings, transpose_b=True)
    rel_edge_pred = tf.gather_nd(all_edge_pred, el)
    edge_pred_loss = tf.losses.sigmoid_cross_entropy(tf.squeeze(w), rel_edge_pred)

    # total
    loss = label_pred_loss + 0.5 * edge_pred_loss

    """
    Summaries
    """
    present_pred_classes = tf.gather(predicted_classes, labelled_verts)
    accuracy = tf.metrics.accuracy(labels=present_labels,
                                   predictions=present_pred_classes,
                                   name='acc_op')
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    """
    Training
    """
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
