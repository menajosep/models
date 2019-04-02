import tensorflow as tf

import reader
import util

from configs import CUDNN, BASIC, BLOCK
import numpy as np


def data_type(use_fp16=False):
    return tf.float16 if use_fp16 else tf.float32


class PTBInput(object):
    """The input data."""

    def __init__(self, config, data, name=None):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
        self.input_data, self.targets = reader.ptb_producer(
            data, batch_size, num_steps, name=name)


class PTBModel(object):
    """The PTB model."""

    def __init__(self, is_training, is_aleatoric, config, input_, num_samples):
        self._is_training = is_training
        self._is_aleatoric = is_aleatoric
        self._input = input_
        self._rnn_params = None
        self._cell = None
        self._logits_sigma = None
        self._sigma_entropy = None
        self.batch_size = input_.batch_size
        self.num_steps = input_.num_steps
        self.embedding_size = config.embedding_size
        self.size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.num_samples = num_samples

        with tf.device("/cpu:0"):
            self._embedding = tf.get_variable(
                "embedding", [self.vocab_size, self.embedding_size], dtype=data_type())
            inputs = tf.nn.embedding_lookup(self._embedding, input_.input_data)

        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)

        output, state = self._build_rnn_graph(inputs, config, is_training)

        if config.tie_embeddings:
            softmax_w = tf.transpose(self._embedding)
            if config.use_projection:
                linear_w = tf.get_variable(
                    "linear_w", [self.size, self.embedding_size], dtype=data_type())
                linear_b = tf.get_variable("linear_b", [self.embedding_size], dtype=data_type())
                output = tf.nn.xw_plus_b(output, linear_w, linear_b)
        else:
            softmax_w = tf.get_variable(
                "softmax_w", [self.size, self.vocab_size], dtype=data_type())
        softmax_b = tf.get_variable("softmax_b", [self.vocab_size], dtype=data_type())
        logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
        self._logits_mu = logits
        # Reshape logits to be a 3-D tensor for sequence loss
        logits = tf.reshape(logits, [self.batch_size, self.num_steps, self.vocab_size])

        if is_aleatoric:
            sigma_w1 = tf.get_variable(
                "sigma_w1", [self.size, self.vocab_size//4], dtype=data_type())
            sigma_b1 = tf.get_variable("sigma_b1", [self.vocab_size//4], dtype=data_type())
            sigma1 = tf.nn.xw_plus_b(output, sigma_w1, sigma_b1)
            sigma1 = tf.nn.selu(sigma1)
            sigma_w2 = tf.get_variable(
                "sigma_w2", [self.vocab_size//4, self.vocab_size//2], dtype=data_type())
            sigma_b2 = tf.get_variable("sigma_b2", [self.vocab_size//2], dtype=data_type())
            sigma2 = tf.nn.xw_plus_b(sigma1, sigma_w2, sigma_b2)
            sigma2 = tf.nn.selu(sigma2)
            sigma_w3 = tf.get_variable(
                "sigma_w3", [self.vocab_size//2, self.vocab_size], dtype=data_type())
            sigma_b3 = tf.get_variable("sigma_b3", [self.vocab_size], dtype=data_type())
            logits_sigma = tf.nn.xw_plus_b(sigma2, sigma_w3, sigma_b3)
            logits_sigma = tf.abs(logits_sigma)
            self._logits_sigma = logits_sigma
            # Reshape logits to be a 3-D tensor for sequence loss
            logits_sigma = tf.reshape(logits_sigma, [self.batch_size, self.num_steps, self.vocab_size])

            one_hot_labels = tf.one_hot(input_.targets, depth=self.vocab_size, dtype=tf.float32)
            self._labels = tf.reshape(input_.targets, [-1])
            crossent, loss, self._probs, self._error, self._mu_entropy, self._voting = \
                self.crossentropy_loss_with_uncert(logits, logits_sigma, one_hot_labels)
        else:
            loss = self.crossentropy_loss(logits, input_.targets)
            crossent = loss
        # Update the cost
        self._loss = tf.reduce_sum(loss)
        self._cost = tf.reduce_sum(crossent)
        self._cost_sigma = tf.reduce_sum(loss)
        self._final_state = state

        if not is_training:
            return

        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        if is_aleatoric:
            tvars = [tvar for tvar in tvars if "sigma" in tvar.name]
        grads, _ = tf.clip_by_global_norm(tf.gradients(self._loss, tvars),
                                          config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.train.get_or_create_global_step())

        self._new_lr = tf.placeholder(
            tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def crossentropy_loss(self, logits, targets):
        num_classes = tf.shape(logits)[2]
        logits_flat = tf.reshape(logits, [-1, num_classes])
        targets = tf.reshape(targets, [-1])
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=targets, logits=logits_flat)
        batch_size = tf.shape(logits)[0]
        sequence_length = tf.shape(logits)[1]
        crossent = tf.reshape(crossent, [batch_size, sequence_length])
        crossent = tf.reduce_mean(crossent, axis=[0])
        return crossent

    def crossentropy_loss_with_uncert(self, logits_mu, logits_sigma, y_true):
        epsilon = 1e-7
        batch_size = tf.shape(logits_mu)[0]
        sequence_length = tf.shape(logits_mu)[1]
        num_classes = tf.shape(logits_mu)[2]
        logits_mu_flat = tf.reshape(logits_mu, [-1, num_classes])
        logits_sigma_flat = tf.reshape(logits_sigma, [-1, num_classes])
        logits_phi = tf.expand_dims(logits_mu_flat, axis=0)
        logits_psi = tf.expand_dims(logits_sigma_flat, axis=0)
        targets = tf.reshape(y_true, [-1, num_classes])
        noise = tf.random_normal((self.num_samples, batch_size*sequence_length, num_classes))
        z = tf.tile(logits_phi, [self.num_samples, 1, 1]) + noise * tf.tile(logits_psi, [self.num_samples, 1, 1])
        z_probs = tf.nn.softmax(z, axis=-1)
        e_probs = tf.reduce_mean(z_probs, axis=0)
        log_probs = tf.log(e_probs)
        sample_xentropy = -(tf.reduce_sum(targets * log_probs, axis=-1))
        sample_xentropy = tf.reshape(sample_xentropy, [batch_size, sequence_length])
        sample_xentropy = tf.reduce_mean(sample_xentropy, axis=0)


        voting = tf.argmax(z, axis=-1, output_type=tf.int32)
        voting = tf.reshape(voting, (lambda shape: (self.num_samples, -1))(tf.shape(voting)))
        voting = tf.one_hot(voting, axis=-1, depth=2)
        voting = tf.reduce_sum(voting, axis=0)
        winner_classes = tf.argmax(logits_mu_flat, axis=1)
        winner_classes = tf.one_hot(winner_classes, axis=-1, depth=2)
        voting = tf.reduce_sum(voting * winner_classes, axis=-1)

        probs = tf.nn.softmax(logits_mu_flat, axis=-1)
        log_probs = tf.log(probs + epsilon)
        cross_entropy = -tf.reduce_sum(targets * log_probs, axis=-1)
        mu_entropy = -tf.reduce_sum(probs * log_probs, axis=-1)
        error = cross_entropy
        cross_entropy = tf.reshape(cross_entropy, [batch_size, sequence_length])
        cross_entropy = tf.reduce_mean(cross_entropy, axis=0)
        return cross_entropy, sample_xentropy, probs, error, mu_entropy, voting

    def _build_rnn_graph(self, inputs, config, is_training):
        return self._build_rnn_graph_lstm(inputs, config, is_training)


    def _get_lstm_cell(self, config, is_training):
        if config.rnn_mode == BASIC:
            return tf.contrib.rnn.BasicLSTMCell(
                config.hidden_size, forget_bias=0.0, state_is_tuple=True,
                reuse=not is_training)
        if config.rnn_mode == BLOCK:
            return tf.contrib.rnn.LSTMBlockCell(
                config.hidden_size, forget_bias=0.0)
        raise ValueError("rnn_mode %s not supported" % config.rnn_mode)

    def _build_rnn_graph_lstm(self, inputs, config, is_training):
        """Build the inference graph using canonical LSTM cells."""

        # Slightly better results can be obtained with forget gate biases
        # initialized to 1 but the hyperparameters of the model would need to be
        # different than reported in the paper.
        def make_cell():
            cell = self._get_lstm_cell(config, is_training)
            if is_training and config.keep_prob < 1:
                cell = tf.contrib.rnn.DropoutWrapper(
                    cell, output_keep_prob=config.keep_prob)
            return cell

        cell = tf.contrib.rnn.MultiRNNCell(
            [make_cell() for _ in range(config.num_layers)], state_is_tuple=True)

        self._initial_state = cell.zero_state(config.batch_size, data_type())
        state = self._initial_state
        inputs = tf.unstack(inputs, num=self.num_steps, axis=1)
        # sze of outputs  (batch_size, num_steps, hidden_size)
        outputs, state = tf.nn.static_rnn(cell, inputs,
                                          initial_state=self._initial_state)
        output = tf.reshape(tf.concat(outputs, 1), [-1, config.hidden_size])
        return output, state

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    def export_ops(self, name):
        """Exports ops to collections."""
        self._name = name
        ops = {util.with_prefix(self._name, "cost"): self._cost}
        if self._is_training:
            ops.update(lr=self._lr, new_lr=self._new_lr, lr_update=self._lr_update)
            if self._rnn_params:
                ops.update(rnn_params=self._rnn_params)
        for name, op in ops.items():
            tf.add_to_collection(name, op)
        self._initial_state_name = util.with_prefix(self._name, "initial")
        self._final_state_name = util.with_prefix(self._name, "final")
        util.export_state_tuples(self._initial_state, self._initial_state_name)
        util.export_state_tuples(self._final_state, self._final_state_name)

    def import_ops(self):
        """Imports ops from collections."""
        if self._is_training:
            self._train_op = tf.get_collection_ref("train_op")[0]
            self._lr = tf.get_collection_ref("lr")[0]
            self._new_lr = tf.get_collection_ref("new_lr")[0]
            self._lr_update = tf.get_collection_ref("lr_update")[0]
            rnn_params = tf.get_collection_ref("rnn_params")
            if self._cell and rnn_params:
                params_saveable = tf.contrib.cudnn_rnn.RNNParamsSaveable(
                    self._cell,
                    self._cell.params_to_canonical,
                    self._cell.canonical_to_params,
                    rnn_params,
                    base_variable_scope="Model/RNN")
                tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, params_saveable)
        self._cost = tf.get_collection_ref(util.with_prefix(self._name, "cost"))[0]
        num_replicas = 1
        self._initial_state = util.import_state_tuples(
            self._initial_state, self._initial_state_name, num_replicas)
        self._final_state = util.import_state_tuples(
            self._final_state, self._final_state_name, num_replicas)

    @property
    def input(self):
        return self._input

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op

    @property
    def initial_state_name(self):
        return self._initial_state_name

    @property
    def final_state_name(self):
        return self._final_state_name

    @property
    def embedding(self):
        return self._embedding

    @property
    def is_aleatoric(self):
        return self._is_aleatoric

    @property
    def is_training(self):
        return self._is_training

    @property
    def logits_mu(self):
        return self._logits_mu

    @property
    def logits_sigma(self):
        return self._logits_sigma

    @property
    def cost_sigma(self):
        return self._cost_sigma

    @property
    def error(self):
        return self._error

    @property
    def probs(self):
        return self._probs

    @property
    def labels(self):
        return self._labels

    @property
    def mu_entropy(self):
        return self._mu_entropy

    @property
    def voting(self):
        return self._voting
