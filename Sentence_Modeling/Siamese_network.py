import tensorflow as tf

class SiameseLSTM(object):
    """
    A LSTM based deep Siamese network
    Uses an character embedding layer, followed by a biLSTM and Energy Loss layer.
    """

    def __init__(self, sequence_length, vocab_processer, embedding_size, hidden_unit_size, l2_reg_lambda, batch_size,
                 embedding_arr):
        # Placeholders for input, output
        self.input_x1 = tf.placeholder(tf.int32, [None, sequence_length], name="input_x1")
        self.mask_x1 = tf.placeholder(tf.float32, [None, sequence_length], name="mask_x1")

        self.input_x2 = tf.placeholder(tf.int32, [None, sequence_length], name="input_x2")
        self.mask_x2 = tf.placeholder(tf.float32, [None, sequence_length], name="mask_x2")
        self.input_y = tf.placeholder(tf.float32, [None], name="input_y")

        self.hidden_n = hidden_unit_size
        self.vocab_processer = vocab_processer

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0, name="l2_loss")

        # Embedding layer
        with tf.name_scope("embedding"):
            self.W = tf.Variable(tf.constant(embedding_arr, dtype=tf.float32), trainable=True, name="W")

            self.embedded_chars1 = tf.nn.embedding_lookup(self.W, self.input_x1)
            self.embedded_chars2 = tf.nn.embedding_lookup(self.W, self.input_x2)

        # Create a convolution + maxpool layer for each filter size
        with tf.name_scope("output"):
            # Siamese Network with the same Bi-LSTM(side, side)
            # self.out1 = self.BiRNN(self.embedded_chars1, self.mask_x1, "side", embedding_size, sequence_length,
            #                        reuse_f=None)
            # self.out2 = self.BiRNN(self.embedded_chars2, self.mask_x2, "side", embedding_size, sequence_length,
            #                        reuse_f=True)

            # Siamese Network with different Bi-LSTMs(side1, side2)
            # out1 shape(None, 20)
            self.out1 = self.BiRNN(self.embedded_chars1, self.mask_x1, "side1", embedding_size, sequence_length)
            self.out2 = self.BiRNN(self.embedded_chars2, self.mask_x2, "side2", embedding_size, sequence_length)

            # cosine distance
            # normalize_a = tf.nn.l2_normalize(self.out1, 1)
            # normalize_b = tf.nn.l2_normalize(self.out2, 1)
            # self.distance = tf.subtract(1.0, abs(
            #     tf.reduce_sum(tf.multiply(normalize_a, normalize_b), axis=1, name="distance")))

            # Euclidean distance : distance shape:(None)

            self.distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(self.out1, self.out2)), 1))

        with tf.name_scope("sentence_embedding"):
            self.representation1 = self.get_Representation(self.embedded_chars1, "side1", embedding_size,
                                                           sequence_length)
            self.representation2 = self.get_Representation(self.embedded_chars2, "side2", embedding_size,
                                                           sequence_length)

            self.representation1 = tf.identity(self.representation1, name="Representation1")
            self.representation2 = tf.identity(self.representation2, name="Representation2")

        with tf.name_scope("loss"):

            self.loss = self.contrastive_loss(self.input_y, self.distance, batch_size)

        with tf.name_scope("accuracy"):
            # predict_label = tf.subtract(1.0, tf.round(self.distance))

            margin = 0.5
            self.predict_label = tf.cast(tf.less(self.distance, margin), "float32", name="prediction")
            # predict_label = tf.subtract(tf.Variable(1.0), tf.round(self.distance))

            correct_predictions = tf.equal(self.predict_label, self.input_y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float32"), name="accuracy")

    def BiRNN(self, x, mask, scope, embedding_size, sequence_length, reuse_f=None):
        n_input = embedding_size
        n_steps = sequence_length

        x = tf.transpose(x, [1, 0, 2])  # (batch_size, n_steps, n_input) => (n_steps, batch_size, n_input)
        # Reshape to (n_steps*batch_size, n_input)
        x = tf.reshape(x, [-1, n_input])
        # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        x = tf.split(x, n_steps, 0)

        with tf.name_scope("fw" + scope), tf.variable_scope("fw" + scope, reuse=reuse_f):
            print(tf.get_variable_scope().name)
            fw_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_n, forget_bias=1.0, state_is_tuple=True)

        with tf.name_scope("bw" + scope), tf.variable_scope("bw" + scope, reuse=reuse_f):
            print(tf.get_variable_scope().name)
            bw_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_n, forget_bias=1.0, state_is_tuple=True)

        with tf.name_scope("fwbw" + scope), tf.variable_scope("fwbw" + scope, reuse=reuse_f):
            outputs, _, output_state_bw = tf.contrib.rnn.static_bidirectional_rnn(fw_cell, bw_cell, x, dtype=tf.float32)

        outputs = tf.stack(outputs)  # list of Tensor(None*(2*hidden)) => steps * None * (2*hidden)

        outputs = tf.transpose(outputs, [1, 0, 2])  # steps * None * (2*hidden) => None * steps * (2 * hidden)
        print(outputs.get_shape().as_list())

        # Use mask
        outputs = outputs * mask[:, :, None]
        # mean pooling to get the vector
        x = tf.reduce_sum(mask, 1)[:, None]
        outputs = tf.reduce_sum(outputs, 1) / x
        print(outputs.get_shape().as_list())
        return outputs

    def get_Representation(self, x, scope, embedding_size, sequence_length):
        n_input = embedding_size
        n_steps = sequence_length

        x = tf.transpose(x, [1, 0, 2])
        # Reshape to (n_steps*batch_size, n_input)
        x = tf.reshape(x, [-1, n_input])
        # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        x = tf.split(x, n_steps, 0)

        with tf.name_scope("fw" + scope), tf.variable_scope("fw" + scope, reuse=True):
            print(tf.get_variable_scope().name)
            fw_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_n, forget_bias=1.0, state_is_tuple=True)
        with tf.name_scope("bw" + scope), tf.variable_scope("bw" + scope, reuse=True):
            print(tf.get_variable_scope().name)
            bw_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_n, forget_bias=1.0, state_is_tuple=True)
        with tf.name_scope("fwbw" + scope), tf.variable_scope("fwbw" + scope, reuse=True):
            # Outputs list contains the depth-concatenated fw and bw vectors for each input.
            # output shape -- [time][batch][cell_fw.output_size + cell_bw.output_size]
            outputs, _, output_state_bw = tf.contrib.rnn.static_bidirectional_rnn(fw_cell, bw_cell, x, dtype=tf.float32)
        return outputs

    def contrastive_loss(self, y, d, batch_size):
        tmp = y * tf.square(d)
        margin = 1
        tmp2 = (1 - y) * tf.square(tf.maximum((margin - d), 0))
        return tf.reduce_sum(tmp + tmp2) / batch_size / 2

        # dis = tf.subtract(tf.Variable(1.0), d)
        # tmp = tf.square(dis) / 4
        # margin = 0.4
        # tmp2 = tf.multiply(tf.cast(tf.less(margin, d), "float"), tf.square(d))
        # return tf.reduce_sum(y * tmp + (1 - y) * tmp2)
