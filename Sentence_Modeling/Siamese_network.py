import tensorflow as tf
import numpy as np
from DLDisambiguation.util.util import getEmbedding


class SiameseLSTM(object):
    """
    A LSTM based deep Siamese network
    Uses an character embedding layer, followed by a biLSTM and Energy Loss layer.
    """

    def BiRNN(self, x, dropout, scope, embedding_size, sequence_length):
        n_input = embedding_size
        n_steps = sequence_length
        n_hidden = 10
        # Prepare data shape to match `bidirectional_rnn` function requirements
        # Current data input shape: (batch_size, n_steps, n_input) (?, seq_len, embedding_size)
        # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

        # Permuting batch_size and n_steps
        x = tf.transpose(x, [1, 0, 2])  # (batch_size, n_steps, n_input) => (n_steps, batch_size, n_input)
        # Reshape to (n_steps*batch_size, n_input)
        x = tf.reshape(x, [-1, n_input])
        # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        x = tf.split(x, n_steps, 0)

        with tf.name_scope("fw" + scope), tf.variable_scope("fw" + scope):
            print(tf.get_variable_scope().name)
            fw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        with tf.name_scope("bw" + scope), tf.variable_scope("bw" + scope):
            print(tf.get_variable_scope().name)
            bw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        with tf.name_scope("bwfw" + scope), tf.variable_scope("bw" + scope):
            outputs, _, output_state_bw = tf.contrib.rnn.static_bidirectional_rnn(fw_cell, bw_cell, x,
                                                                                      dtype=tf.float32)
        return outputs[-1]

    def get_Representation(self, x, dropout, scope, embedding_size, sequence_length):
        n_input = embedding_size
        n_steps = sequence_length
        n_hidden = 10

        x = tf.transpose(x, [1, 0, 2])
        # Reshape to (n_steps*batch_size, n_input)
        x = tf.reshape(x, [-1, n_input])
        # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        x = tf.split(x, n_steps, 0)

        # Bi-LSTM
        with tf.name_scope("fw" + scope), tf.variable_scope("fw" + scope, reuse=True):
            print(tf.get_variable_scope().name)
            fw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        with tf.name_scope("bw" + scope), tf.variable_scope("bw" + scope, reuse=True):
            print(tf.get_variable_scope().name)
            bw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        with tf.name_scope("fwbw" + scope), tf.variable_scope("bw" + scope, reuse=True):
            # Outputs list contains the depth-concatenated fw and bw vectors for each input.
            # output shape -- [time][batch][cell_fw.output_size + cell_bw.output_size]
            outputs, _, output_state_bw = tf.contrib.rnn.static_bidirectional_rnn(fw_cell, bw_cell, x, dtype=tf.float32)
        return outputs

    def contrastive_loss(self, y, d, batch_size):
        tmp = y * tf.square(d)
        tmp2 = (1 - y) * tf.square(tf.maximum((1 - d), 0))
        return tf.reduce_sum(tmp + tmp2) / batch_size / 2

    def getEmbedding_Construction(self, vocab, embedding, processer):
        # Extract word:id mapping from the object.
        vocab_dict = processer.vocabulary_._mapping
        words_s = set(vocab_dict.keys())

        # words_index = [i for i,e in enumerate(vocab) if e.decode("utf-8") in words_s]
        words_index = []
        vocab_set = set(vocab)
        last_index = len(vocab) - 1

        for i in words_s:
            character_u = i.encode("utf-8")
            if character_u in vocab_set:
                words_index.append(vocab_dict[i])
            else:
                # for unkown word, give default value of key <UNK>
                print(character_u)
                words_index.append(last_index)
                # print(embedding[last_index])

        # words_index = [vocab_dict[i] for i in words_s if i.encode("utf-8") in set(vocab)]
        res = embedding[words_index]
        res = np.asarray(res, dtype="float")
        return res

    def loadEmbeddingLayer(self, sess):

        embedding_init = self.W.assign(self.embedding_placeholder)

        # load in word2vec model
        vocab, vocab_size, embedding_dim, embedding = getEmbedding("../data/word2vec/character_model.txt")

        embedding_construct = self.getEmbedding_Construction(vocab, embedding, self.vocab_processer)
        sess.run(embedding_init, feed_dict={self.embedding_placeholder: embedding_construct})

    def __init__(
            self, sess, sequence_length, vocab_processer, embedding_size, hidden_units, l2_reg_lambda, batch_size):
        # Placeholders for input, output and dropout
        self.input_x1 = tf.placeholder(tf.int32, [None, sequence_length], name="input_x1")
        self.input_x2 = tf.placeholder(tf.int32, [None, sequence_length], name="input_x2")
        self.input_y = tf.placeholder(tf.float32, [None], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        self.vocab_processer = vocab_processer
        vocab_size = len(vocab_processer.vocabulary_)

        # Keeping track of l2 regularization loss (optional)
        # l2_loss = tf.constant(0.0, name="l2_loss")

        # Embedding layer
        with tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                trainable=True, name="W")
            self.embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_size])

            self.embedded_chars1 = tf.nn.embedding_lookup(self.W, self.input_x1)
            # self.embedded_chars_expanded1 = tf.expand_dims(self.embedded_chars1, -1)
            self.embedded_chars2 = tf.nn.embedding_lookup(self.W, self.input_x2)
            # self.embedded_chars_expanded2 = tf.expand_dims(self.embedded_chars2, -1)

        # Create a convolution + maxpool layer for each filter size
        with tf.name_scope("output"):
            self.out1 = self.BiRNN(self.embedded_chars1, self.dropout_keep_prob, "side1", embedding_size,
                                   sequence_length)
            self.out2 = self.BiRNN(self.embedded_chars2, self.dropout_keep_prob, "side2", embedding_size,
                                   sequence_length)

            # cosine distance
            normalize_a = tf.nn.l2_normalize(self.out1, 1)
            normalize_b = tf.nn.l2_normalize(self.out2, 1)
            self.distance = tf.reduce_sum(tf.multiply(normalize_a, normalize_b), axis=1, name="distance")
            print self.distance.get_shape().as_list()

            # self.distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(self.out1, self.out2)), 1, keep_dims=True))
            # # self.distance = tf.div(self.distance,
            # #                        tf.add(tf.sqrt(tf.reduce_sum(tf.square(self.out1), 1, keep_dims=True)),
            # #                               tf.sqrt(tf.reduce_sum(tf.square(self.out2), 1, keep_dims=True))))
            # self.distance = tf.reshape(self.distance, [-1], name="distance")

        with tf.name_scope("sentence_embedding"):
            self.representation1 = self.get_Representation(self.embedded_chars1, self.dropout_keep_prob, "side1",
                                                           embedding_size, sequence_length)
            self.representation2 = self.get_Representation(self.embedded_chars2, self.dropout_keep_prob, "side2",
                                                           embedding_size, sequence_length)

            self.representation1 = tf.identity(self.representation1, name="Representation1")
            self.representation2 = tf.identity(self.representation2, name="Representation2")

        with tf.name_scope("loss"):
            self.loss = self.contrastive_loss(self.input_y, self.distance, batch_size)

        with tf.name_scope("accuracy"):
            predict_label = tf.subtract(tf.Variable(1.0), tf.round(self.distance))
            correct_predictions = tf.equal(predict_label, self.input_y)
            # correct_predictions = tf.equal(self.distance, self.input_y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
