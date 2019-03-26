# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Example / benchmark for building a PTB LSTM model.

Trains the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329

There are 3 supported model configurations:
===========================================
| config | epochs | train | valid  | test
===========================================
| small  | 13     | 37.99 | 121.39 | 115.91
| medium | 39     | 48.45 |  86.16 |  82.07
| large  | 55     | 37.87 |  82.62 |  78.29
The exact results may vary depending on the random initialization.

The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size
- rnn_mode - the low level implementation of lstm cell: one of CUDNN,
             BASIC, or BLOCK, representing cudnn_lstm, basic_lstm, and
             lstm_block_cell classes.

The data required for this example is in the data/ dir of the
PTB dataset from Tomas Mikolov's webpage:

$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz

To run:

$ python ptb_word_lm.py --data_path=simple-examples/data/

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import dill
import numpy as np
import tensorflow as tf

import reader

from models import PTBInput, PTBModel
from configs import *


flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", None,
                    "Where the training/test data is stored.")
flags.DEFINE_string("save_path", None,
                    "Model output directory.")
flags.DEFINE_bool("is_training", True,
                  "tells if the model must be trained or restore")
flags.DEFINE_bool("is_aleatoric", False,
                  "tells if the model must be trained for aleatoric uncertainty")
flags.DEFINE_string("restore_path", None,
                    "Model input directory.")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")
flags.DEFINE_string("rnn_mode", None,
                    "The low level implementation of lstm cell: one of CUDNN, "
                    "BASIC, and BLOCK, representing cudnn_lstm, basic_lstm, "
                    "and lstm_block_cell classes.")
flags.DEFINE_integer("num_samples", 100,
                     "Number of sampls for aleatoric ucnertainty")
FLAGS = flags.FLAGS


def run_epoch(session, model, eval_op=None, verbose=False, is_aleatoric=False, is_training=True):
    """Runs the model on the given data."""
    start_time = time.time()
    costs = 0.0
    costs_sigma = 0.0
    iters = 0
    logits_mu = None
    logits_sigma = None
    sigma_entropies = None
    errors = None
    baselines = None
    embedding = None
    labels = None
    predictions = None
    votings = None
    mu_entropies = None
    state = session.run(model.initial_state)

    fetches = {
        "cost": model.cost,
        "cost_sigma": model.cost_sigma,
        "final_state": model.final_state,
        "embedding": model.embedding,
        "logits_mu": model.logits_mu,
        "probs": model.probs,
        "labels": model.labels

    }
    if is_aleatoric:
        fetches["logits_sigma"] = model.logits_sigma
        fetches["sigma_entropy"] = model.sigma_entropy
        fetches["error"] = model.error
        fetches["voting"] = model.voting
        fetches["mu_entropy"] = model.mu_entropy

    if eval_op is not None:
        fetches["eval_op"] = eval_op

    for step in range(model.input.epoch_size):
        feed_dict = {}
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

        vals = session.run(fetches, feed_dict)
        cost = vals["cost"]
        cost_sigma = vals["cost_sigma"]
        state = vals["final_state"]
        embedding = vals["embedding"]
        if is_aleatoric:
            logits_mu = vals["logits_mu"]
            logits_sigma = vals["logits_sigma"]
            if not is_training:
                batch_probs = vals["probs"]
                batch_prob_argmax = batch_probs.argmax(axis=-1)
                batch_winner_probs = np.array([prob[batch_prob_argmax[i]] for i, prob in enumerate(batch_probs)])
                batch_baseline = np.minimum(1-batch_winner_probs, batch_winner_probs)
                if labels is None:
                    labels = vals["labels"]
                else:
                    labels = np.append(labels, vals["labels"], axis=0)
                if predictions is None:
                    predictions = batch_prob_argmax
                else:
                    predictions = np.append(predictions, batch_prob_argmax, axis=0)
                if baselines is None:
                    baselines = batch_baseline
                else:
                    baselines = np.append(baselines, batch_baseline, axis=0)
                if sigma_entropies is None:
                    sigma_entropies = vals["sigma_entropy"]
                else:
                    sigma_entropies = np.append(sigma_entropies, vals["sigma_entropy"], axis=0)
                if votings is None:
                    votings = vals["voting"]
                else:
                    votings = np.append(votings, vals["voting"], axis=0)
                if mu_entropies is None:
                    mu_entropies = vals["mu_entropy"]
                else:
                    mu_entropies = np.append(mu_entropies, vals["mu_entropy"], axis=0)
                if errors is None:
                    errors = vals["error"]
                else:
                    errors = np.append(errors, vals["error"], axis=0)
        costs += cost
        costs_sigma += cost_sigma
        iters += model.input.num_steps

        if verbose and step % (model.input.epoch_size // 10) == 10:
            print("%.3f perplexity: %.3f perplexity_sigma %.3f speed: %.0f wps" %
                  (step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
                   np.exp(costs_sigma / iters),
                   iters * model.input.batch_size /
                   (time.time() - start_time)))

    return np.exp(costs / iters), np.exp(costs_sigma / iters), \
           logits_mu, logits_sigma, \
           embedding, \
           baselines, errors, sigma_entropies, labels, predictions, votings, mu_entropies


def get_config():
    """Get model config."""
    if FLAGS.model == "small":
        config = SmallConfig()
    elif FLAGS.model == "medium":
        config = MediumConfig()
    elif FLAGS.model == "large":
        config = LargeConfig()
    elif FLAGS.model == "baseline":
        config = LargeConfig()
    elif FLAGS.model == "test":
        config = TestConfig()
    elif FLAGS.model == "newtest":
        config = NewTestConfig()
    elif FLAGS.model == "baselinetied":
        config = TiedLargeConfig()
    elif FLAGS.model == "baselinebayes":
        config = BayesMediumConfig()
    elif FLAGS.model == "nontied":
        config = NewLargeConfig()
    elif FLAGS.model == "tied":
        config = NewTiedLargeConfig()
    elif FLAGS.model == "tiedl":
        config = NewTiedLLargeConfig()
    else:
        raise ValueError("Invalid model: %s", FLAGS.model)
    if FLAGS.rnn_mode:
        config.rnn_mode = FLAGS.rnn_mode
    return config


def main(_):
    if not FLAGS.data_path:
        raise ValueError("Must set --data_path to PTB data directory")

    if FLAGS.save_path:
        tf.gfile.MakeDirs(FLAGS.save_path)

    raw_data = reader.ptb_raw_data(FLAGS.data_path)
    train_data, valid_data, test_data, _ = raw_data

    config = get_config()
    eval_config = get_config()
    eval_config.batch_size = 1
    eval_config.num_steps = 1

    with tf.Graph().as_default() as graph:
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)

        with tf.name_scope("Train"):
            train_input = PTBInput(config=config, data=train_data, name="TrainInput")
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                m = PTBModel(is_training=True, is_aleatoric=FLAGS.is_aleatoric,
                             config=config, input_=train_input, num_samples=FLAGS.num_samples)
            tf.summary.scalar("Training Loss", m.cost)
            tf.summary.scalar("Learning Rate", m.lr)

        with tf.name_scope("Valid"):
            valid_input = PTBInput(config=config, data=valid_data, name="ValidInput")
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mvalid = PTBModel(is_training=False, is_aleatoric=FLAGS.is_aleatoric,
                                  config=config, input_=valid_input, num_samples=FLAGS.num_samples)
            tf.summary.scalar("Validation Loss", mvalid.cost)

        with tf.name_scope("Test"):
            test_input = PTBInput(
                config=eval_config, data=test_data, name="TestInput")
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mtest = PTBModel(is_training=False, is_aleatoric=FLAGS.is_aleatoric,
                                 config=eval_config,
                                 input_=test_input,
                                 num_samples=FLAGS.num_samples)

        # Add ops to save and restore all the variables.
        vars = {var.name[:-2]: var for var in tf.global_variables() if "sigma" not in var.name}
        saver = tf.train.Saver(vars)

        with tf.Session(graph=graph) as session:
            session.run(tf.global_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(session)
            if FLAGS.is_training:
                if FLAGS.is_aleatoric:
                    saver.restore(session, FLAGS.restore_path)
                for i in range(config.max_max_epoch):
                    lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
                    m.assign_lr(session, config.learning_rate * lr_decay)

                    print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
                    train_perplexity, train_perplexity_sigma, logits_mu, logits_sigma, _, _, _, _, _, _, _, _ = \
                        run_epoch(session, m, eval_op=m.train_op,verbose=True, is_aleatoric=FLAGS.is_aleatoric,
                                  is_training=True)
                    print("Epoch: %d Train Perplexity: %.3f Perplexity sigma: %.3f" %
                          (i + 1, train_perplexity, train_perplexity_sigma))
                    if FLAGS.is_aleatoric:
                        print("Epoch: mu: %s sigma %s" %
                              (str(logits_mu[0][:10]), str(logits_sigma[0][:10])))
                    valid_perplexity, valid_perplexity_sigma, logits_mu, logits_sigma, _, _, _, _, _, _, _, _ = \
                        run_epoch(session, mvalid, is_aleatoric=FLAGS.is_aleatoric, is_training=True)
                    print("Epoch: %d Valid Perplexity: %.3f Perplexity sigma: %.3f" %
                          (i + 1, valid_perplexity, valid_perplexity_sigma
                           ))
                    if FLAGS.is_aleatoric:
                        print("Epoch: mu: %s sigma %s" %
                              (str(logits_mu[0][:10]), str(logits_sigma[0][:10])))
            else:
                saver.restore(session, FLAGS.restore_path)

            test_perplexity, test_perplexity_sigma, \
            logits_mu, logits_sigma, embedding, baselines, \
            errors, sigma_entropies, labels, \
            predictions, votings, mu_entropies = run_epoch(session, mtest,is_aleatoric=FLAGS.is_aleatoric, is_training=False)
            print("Test Perplexity: %.3f Perplexity sigma: %.3f" %
                  (test_perplexity, test_perplexity_sigma))
            if FLAGS.is_aleatoric:
                print("Epoch: mu: %s sigma %s" %
                      (str(logits_mu[0][:10]), str(logits_sigma[0][:10])))

            if FLAGS.save_path:
                print("Saving model to %s." % FLAGS.save_path + '/model')
                save_path = saver.save(session, FLAGS.save_path + '/model')
                print("Saved model to %s." % save_path)
                print("Saving embeddings to %s." % FLAGS.save_path)
                with open(os.path.join(FLAGS.save_path, "embeddings.p"), "wb") as outputfile:
                    dill.dump(embedding, outputfile)
                print("Saved embeddings to %s." % save_path)
                if FLAGS.is_aleatoric:
                    print("Saving sigma entropies and mu errors to %s." % FLAGS.save_path)
                    with open(os.path.join(FLAGS.save_path, "aleatoric_results.p"), "wb") as outputfile:
                        dill.dump({
                            "sigma_entropies": sigma_entropies,
                            "errors": errors,
                            "baselines": baselines,
                            "labels": labels,
                            "predictions": predictions,
                            "voting": votings,
                            "mu_entropies": mu_entropies
                        }, outputfile)
                    print("Saved sigma entropies and mu errors to %s." % save_path)
            coord.request_stop()
            coord.join(threads, ignore_live_threads=True)


if __name__ == "__main__":
    tf.app.run()
