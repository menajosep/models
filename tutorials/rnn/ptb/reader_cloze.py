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


"""Utilities for parsing PTB text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys
import random

import numpy as np
import tensorflow as tf

Py3 = sys.version_info[0] == 3


def _read_words(filename):
    with tf.gfile.GFile(filename, "r") as f:
        if Py3:
            return f.read().replace("\n", "<eos>").split()
        else:
            return f.read().decode("utf-8").replace("\n", "<eos>").split()


def _build_vocab(filename):
    data = _read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    words = list(words)
    words.append('<mask>')
    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id


def _file_to_word_ids(filename, word_to_id, min_sentence_length, max_sentence_length):
    raw_data = _read_words(filename)
    buffer = []
    sentences = []
    for word in raw_data:
        buffer.append(word)
        if word == '<eos>':
            if min_sentence_length < len(buffer) < max_sentence_length:
                data = [word_to_id[word] for word in buffer if word in word_to_id]
                if len(buffer) < max_sentence_length:
                    padding = [word_to_id['<eos>']] * (max_sentence_length - len(buffer))
                    data.extend(padding)

                sentences.append(data)
            buffer = []

    return sentences


def ptb_raw_data(data_path=None, min_sentence_length=1, max_sentence_length=100):
    """Load PTB raw data from data directory "data_path".

    Reads PTB text files, converts strings to integer ids,
    and performs mini-batching of the inputs.

    The PTB dataset comes from Tomas Mikolov's webpage:

    http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

    Args:
      data_path: string path to the directory where simple-examples.tgz has
        been extracted.

    Returns:
      tuple (train_data, valid_data, test_data, vocabulary)
      where each of the data objects can be passed to PTBIterator.
    """

    train_path = os.path.join(data_path, "ptb.train.txt")
    valid_path = os.path.join(data_path, "ptb.valid.txt")
    test_path = os.path.join(data_path, "ptb.test.txt")

    word_to_id = _build_vocab(train_path)
    train_data = _file_to_word_ids(train_path, word_to_id, min_sentence_length, max_sentence_length)
    valid_data = _file_to_word_ids(valid_path, word_to_id, min_sentence_length, max_sentence_length)
    test_data = _file_to_word_ids(test_path, word_to_id, min_sentence_length, max_sentence_length)
    vocabulary = len(word_to_id)
    return train_data, valid_data, test_data, vocabulary, word_to_id


def get_mask_index(sentence):
    return random.randint(0, len(sentence)-1)


def ptb_producer(raw_data, batch_size, num_steps, word_to_id):
    """Iterate on the raw PTB data.

    This chunks up raw_data into batches of examples and returns  these batches.

    Args:
      raw_data: one of the raw data outputs from ptb_raw_data.
      batch_size: int, the batch size.
      num_steps: int, the number of unrolls.

    Returns:
      Am array of [batch_size, num_steps]. The second element
      of the tuple is the index of the masked word

    Raises:
      tf.errors.InvalidArgumentError: if batch_size or num_steps are too high.
    """
    x = []
    y = []
    n_batches = len(raw_data) // batch_size
    for sentence in raw_data:
        mask_index = get_mask_index(sentence)
        current_label = sentence[mask_index]
        sentence[mask_index] = word_to_id['<mask>']
        y.append(current_label)
        x.append(sentence)
    x = np.array(x)
    x = x[:n_batches*batch_size]
    x = np.reshape(x, [n_batches, batch_size, num_steps])
    y = np.array(y)
    y = y[:n_batches * batch_size]
    y = np.reshape(y, [n_batches, batch_size])
    return x, y
