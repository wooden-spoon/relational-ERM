import tensorflow as tf

from . import metrics


def _make_metrics(labels, predictions, weights):
    assert weights is not None

    accuracy = tf.metrics.accuracy(
        labels=labels,
        predictions=predictions,
        weights=tf.expand_dims(weights, -1))

    precision = tf.metrics.precision(
        labels=labels,
        predictions=predictions,
        weights=tf.expand_dims(weights, -1))

    recall = tf.metrics.recall(
        labels=labels,
        predictions=predictions,
        weights=tf.expand_dims(weights, -1))

    macro_f1 = metrics.macro_f1(
        labels=labels,
        predictions=predictions,
        weights=weights)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'macro_f1': macro_f1
    }


def _make_dataset_summaries(features, mode):
    """ Make summaries for dataset (number of edges and vertices seen so far).

    By default, we only update those during training (as they represent the number
    of training samples seen).

    Parameter
    ---------
    features: the features passed into the estimator.
    mode: the estimator mode
    """
    if mode != tf.estimator.ModeKeys.TRAIN:
        return

    with tf.variable_scope(None, 'dataset_summaries'):
        total_count_vertex = tf.get_variable('total_count_vertex', shape=[], dtype=tf.int64,
                                             initializer=tf.zeros_initializer(), trainable=False)
        total_count_edges = tf.get_variable('total_count_edges', shape=[], dtype=tf.int64,
                                            initializer=tf.zeros_initializer(), trainable=False)

        update_vertex_count = total_count_vertex.assign_add(
            tf.shape(features['vertex_index'], out_type=tf.int64)[0])
        update_edge_count = total_count_edges.assign_add(
            tf.shape(features['edge_list'], out_type=tf.int64)[0])

        with tf.control_dependencies([update_vertex_count, update_edge_count]):
            tf.summary.scalar('total_edges', total_count_edges, family='dataset')
            tf.summary.scalar('total_vertex', total_count_vertex, family='dataset')


def _make_label_prediction_summaries(present_labels, present_pred_labels, split):
    """ Make summaries for label prediction task.

    Parameter
    ---------
    present_labels:  the labels present in the graph.
    present_pred_labels: the predicted labels present in the graph.
    split: for present labels, whether they are censored for testing.
    """
    # split == 1 indicates insample, wherease split == 0 indicates out of sample.
    # split == -1 denotes fake padded values.
    split_insample = tf.expand_dims(tf.to_float(tf.equal(split, 1)), -1)
    split_outsample = tf.expand_dims(tf.to_float(tf.equal(split, 0)), -1)

    accuracy_batch_insample = metrics.batch_accuracy(
        present_labels, present_pred_labels, split_insample,
        name='accuracy_insample_batch')
    kappa_batch_insample = metrics.batch_kappa(
        present_labels, present_pred_labels, split_insample,
        name='kappa_insample_batch'
    )
    accuracy_batch_outsample = metrics.batch_accuracy(
        present_labels, present_pred_labels, split_outsample,
        name='accuracy_outsample_batch'
    )
    kappa_batch_outsample = metrics.batch_kappa(
        present_labels, present_pred_labels, split_outsample,
        name='kappa_outsample_batch'
    )
    tf.summary.scalar('accuracy_batch_in', accuracy_batch_insample)
    tf.summary.scalar('accuracy_batch_out', accuracy_batch_outsample)
    tf.summary.scalar('kappa_batch_in', kappa_batch_insample)
    tf.summary.scalar('kappa_batch_out', kappa_batch_outsample)


def _get_value(value_or_fn):
    if callable(value_or_fn):
        return value_or_fn()
    else:
        return value_or_fn


def _default_embedding_optimizer():
    # embedding optimization


    # word2vec decays linearly to a min learning rate (default: 0.0001), decreasing each "epoch"
    # however, node2vec and deepwalk run only 1 "epoch" each

    # learning_rate = tf.train.polynomial_decay(
    #     10.,
    #     global_step,
    #     100000,
    #     end_learning_rate=0.0001,
    #     power=1.0,
    #     cycle=False,
    #     name="Word2Vec_decay"
    # )

    # gensim word2vec default learning rate is 0.025
    return tf.train.GradientDescentOptimizer(learning_rate=0.025)


def _default_global_optimizer():
    # return tf.train.RMSPropOptimizer(learning_rate=5e-4, momentum=0.9)
    global_step = tf.train.get_or_create_global_step()
    # learning_rate = tf.train.polynomial_decay(
    #     10.,
    #     global_step,
    #     1000000,
    #     end_learning_rate=0.01,
    #     power=1.0,
    #     cycle=False,
    #     name="global_linear_decay"
    # )
    learning_rate = 1.
    return tf.train.GradientDescentOptimizer(learning_rate)


def _make_polyak_averaging(embeddings, features, label_logits, mode, polyak, make_label_logits, params):
    batch_size = params['batch_size']
    decay = 0.99

    if batch_size is not None:
        #  Adjust decay for batch size to take into account the minibatching.
        decay = decay ** batch_size

    label_ema = tf.train.ExponentialMovingAverage(decay=decay)
    if polyak:
        # predict logits by replacing the model params by a moving average
        def label_ema_getter(getter, name, *args, **kwargs):
            var = getter(name, *args, **kwargs)
            ema_var = label_ema.average(var)
            return ema_var  # if ema_var else var

        # create the running average variable
        label_ema_op = label_ema.apply(tf.global_variables("label_logits"))
        with tf.control_dependencies([label_ema_op]):
            with tf.variable_scope("label_logits", reuse=True, custom_getter=label_ema_getter):
                label_logits_predict = make_label_logits(embeddings, features, mode, params)
    else:
        # no polyak averaging; default behaviour
        label_logits_predict = label_logits
        label_ema_op = tf.no_op(name='no_polyak_averaging')

    return label_ema_op, label_logits_predict


def _make_embedding_variable(params):
    embedding_variable_name = 'input_layer/vertex_index_embedding/embedding_weights'

    regularizer = tf.contrib.layers.l2_regularizer(scale=0.0)

    all_embeddings = tf.get_variable(
        embedding_variable_name,
        shape=[params['num_vertices'], params['embedding_dim']],
        dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=1 / params['embedding_dim']),
        # regularizer=regularizer,
        trainable=params.get('embedding_trainable', True))
    if params.get('embedding_checkpoint', None) is not None:
        tf.train.init_from_checkpoint(
            params['embedding_checkpoint'],
            {embedding_variable_name: all_embeddings})
    return all_embeddings


def make_node_classifier(make_label_logits,
                         make_edge_logits,
                         make_label_pred_loss,
                         make_edge_pred_loss,
                         embedding_optimizer=None,
                         global_optimizer=None,
                         polyak=True,
                         pos_only_labels=True):
    """ Creates a node classifier function from various parts.

    Parameters
    ----------
    make_label_logits: function (embeddings, features, mode, params) -> (logits),
        which computes the label logits for for each node.
    make_edge_logits: function (embeddings, features, edge_list, edge_weights, params) -> (label_logits),
        which computes the logits for each pair in edge_list.
    make_label_pred_loss: function (label_logits, present_labels) -> (losses),
        which computes the label prediction loss.
    make_edge_pred_loss: function (embeddings, n_vert, el, w, params) -> (losses),
        which computes the edge prediction loss.
    embedding_optimizer: the optimizer (or a nullary function creating the optimizer) to use for the embedding variables.
    global_optimizer: the optimizer (or a nullary function creating the optimizer) to use for the global variables.
    polyak: bool, default True. If true, label predictions are made using an exponentially weighted moving average of
        the global variables
    pos_only_labels: bool, default False. If true, label predictions are trained using only vertices from the positive
        sample

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
        vertex_index = features['vertex_index']

        all_embeddings = _make_embedding_variable(params)

        vertex_embedding_shape = tf.concat(
            [tf.shape(vertex_index), [params['embedding_dim']]], axis=0,
            name='vertex_embedding_shape')

        # We flatten the vertex index prior to extracting embeddings
        # to maintain compatibility with the input columns.
        embeddings = tf.nn.embedding_lookup(all_embeddings, tf.reshape(vertex_index, [-1]))
        embeddings = tf.reshape(embeddings, vertex_embedding_shape, name='vertex_embeddings_batch')

        # Vertex Label Predictions
        present_labels = labels['labels']
        split = labels['split']

        if pos_only_labels:
            vert_is_positive = features['is_positive']
            split = tf.where(tf.equal(vert_is_positive,1), split, -tf.ones_like(split))

        with tf.variable_scope("label_logits"):
            label_logits = make_label_logits(embeddings, features, mode, params)

        # polyak averaging
        label_ema_op, label_logits_predict = _make_polyak_averaging(
            embeddings, features, label_logits, mode, polyak, make_label_logits, params)

        predicted_labels = tf.cast(tf.greater(label_logits_predict, 0.), label_logits.dtype)

        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                'class_ids': predicted_labels,
                'probabilities': tf.nn.sigmoid(label_logits_predict),
                'label_logits': label_logits_predict,
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        # label loss
        with tf.name_scope('label_loss', values=[label_logits, present_labels, split]):
            label_pred_loss = make_label_pred_loss(
                label_logits, present_labels,
                tf.maximum(split, 0))  # clip the split, as -1 represents padded values.

            label_pred_size = tf.shape(label_logits)[-1]
            label_pred_loss_normalized = tf.divide(label_pred_loss, tf.to_float(label_pred_size))

        # label logits and DeepWalk style prediction
        present_logits = label_logits_predict
        present_pred_labels = metrics.oracle_predictions(present_labels, present_logits)

        if mode == tf.estimator.ModeKeys.EVAL:
            # Metrics
            estimator_metrics = {}

            with tf.variable_scope('metrics_insample'):
                estimator_metrics.update({
                    k + '_insample': v
                    for k, v in _make_metrics(
                        present_labels,
                        present_pred_labels,
                        split).items()
                })

            with tf.variable_scope('metrics_outsample'):
                estimator_metrics.update({
                    k + '_outsample': v
                    for k, v in _make_metrics(
                        present_labels,
                        present_pred_labels,
                        (1 - split)).items()
                })

            return tf.estimator.EstimatorSpec(
                mode, loss=label_pred_loss, eval_metric_ops=estimator_metrics)


        # subgraph structure
        edge_list = features['edge_list']
        weights = features['weights']  # should be {0., 1.}
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
        tf.summary.scalar('label_loss_normalized', label_pred_loss_normalized, family='loss')
        tf.summary.scalar('edge_loss', edge_pred_loss, family='loss')
        tf.summary.scalar('edge_loss_normalized', edge_pred_loss_normalized, family='loss')
        tf.summary.scalar('regularization_loss', reg_loss, family='loss')

        # Summaries
        _make_label_prediction_summaries(present_labels, present_pred_labels, split)

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

        # gradient updates
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
                basic_train_op = tf.group(embedding_update, global_update)

            if polyak:
                # update moving average of parameters after each gradient step
                label_ema_op._add_control_input(basic_train_op)
                train_op = label_ema_op
            else:
                train_op = basic_train_op

            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    return node_classifier
