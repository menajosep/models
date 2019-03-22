from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import dill
import numpy as np
import tensorflow as tf

import reader_cloze

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("data_path", None,
                    "Where the training/test data is stored.")
flags.DEFINE_string("save_path", None,
                    "Model output directory.")
flags.DEFINE_string("embedding_path", None,
                    "file with the learned embedddings.")
FLAGS = flags.FLAGS


def load_embeddings(word_to_id, embedding_path):
    word2vec_word_to_vec = {}
    header = True
    with open(embedding_path, "r") as embedding_file:
        for line in embedding_file:
            if header:
                header = False
            else:
                embedding_line = line.split(" ", 1)
                word2vec_word_to_vec[embedding_line[0]] = np.fromstring(embedding_line[1].replace("\n", ""), sep=" ")
    return None


def main(_):
    if not FLAGS.data_path:
        raise ValueError("Must set --data_path to PTB data directory")
    if not FLAGS.embedding_path:
        raise ValueError("Must set --embedding_path to embeddings file")

    raw_data = reader_cloze.ptb_raw_data(FLAGS.data_path)
    train_data, valid_data, test_data, _, word_to_id = raw_data

    embeddings = load_embeddings(word_to_id, FLAGS.embedding_path)

    with open(os.path.join(FLAGS.save_path, "word2vec.p"), "wb") as outputfile:
        dill.dump(embeddings, outputfile)




if __name__ == "__main__":
    tf.app.run()
