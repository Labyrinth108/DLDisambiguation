#!/usr/bin/env python
# encoding=utf-8
import numpy as np
import tensorflow as tf


class MultiGranModel(object):
    def _conv(self, name, in_, ksize, reuse=False):
        num_filters = ksize[3]

        with tf.variable_scope(name, reuse=reuse) as scope:
            # different CNN for different views
            # W = tf.get_variable("weights", ksize, initializer=tf.contrib.layers.xavier_initializer())
            W = tf.Variable(tf.truncated_normal(ksize, stddev=0.1), name="W")
            biases = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")

            # same CNN for different views
            # W = tf.get_variable("weights", ksize, initializer=tf.contrib.layers.xavier_initializer())
            # W = tf.get_variable("weights", ksize, initializer=tf.truncated_normal_initializer(stddev=0.1))
            # biases = tf.get_variable("biases", [num_filters], initializer=tf.constant_initializer(0.1))

            conv = tf.nn.conv2d(in_, W, strides=[1, 1, 1, 1], padding="VALID")
            h = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope.name)

        return h

    def _maxpool(self, name, in_, ksize, strides):
        pool = tf.nn.max_pool(in_, ksize=ksize, strides=strides, padding='VALID', name=name)
        print name, pool.get_shape().as_list()
        return pool

    def __init__(self, max_len, filter_sizes, pool_sizes, num_filters, l2_reg_lambda=0.0, type_CNN=2):
        channel_num = 4

        # Placeholders for input, output and dropout
        self.input_tensor = tf.placeholder(tf.float32, [None, max_len, max_len, channel_num], name="input_tensor")
        self.input_y = tf.placeholder(tf.float32, [None, 2], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []

        input_tensor = tf.expand_dims(self.input_tensor, 4)  # N x W x H x V  => N x W x H x V x C
        input_tensor = tf.transpose(input_tensor, perm=[3, 0, 1, 2, 4])  # N x W x H x V x C =>  V x N x W x H x C

        if type_CNN == 1:
            filter_shape1 = [filter_sizes[0], filter_sizes[1], 4, num_filters / 2]
            p_size1 = [1, 2, 2, 1]
            filter_shape2 = [filter_sizes[2], filter_sizes[3], num_filters / 2, num_filters]
            p_size2 = [1, 2, 2, 1]

            conv1 = self._conv("conv1", self.input_tensor, filter_shape1)
            pool1 = self._maxpool('pool1', conv1, ksize=p_size1, strides=[1, 1, 1, 1])
            conv2 = self._conv('conv2', pool1, filter_shape2)
            pool2 = self._maxpool('pool2', conv2, ksize=p_size2, strides=[1, 1, 1, 1])

            dim = np.prod(pool2.get_shape().as_list()[1:])
            y = tf.reshape(pool2, [-1, dim])
        else:
            for i in range(channel_num):
                # set reuse True for i > 0, for weight-sharing
                reuse_f = (i != 0)
                view = tf.gather(input_tensor, i)  # N x W x H x C

                filter_shape1 = [filter_sizes[0], filter_sizes[1], 1, num_filters / 2]
                p_size1 = [1, pool_sizes[0], pool_sizes[1], 1]

                conv1 = self._conv('conv1', view, filter_shape1, reuse=reuse_f)
                pool1 = self._maxpool('pool1', conv1, ksize=p_size1, strides=[1, 1, 1, 1])

                if len(filter_sizes) == 4:
                    filter_shape2 = [filter_sizes[2], filter_sizes[3], num_filters / 2, num_filters]
                    p_size2 = [1, pool_sizes[2], pool_sizes[3], 1]

                    conv2 = self._conv('conv2', pool1, filter_shape2, reuse=reuse_f)
                    pool2 = self._maxpool('pool2', conv2, ksize=p_size2, strides=[1, 1, 1, 1])

                    dim = np.prod(pool2.get_shape().as_list()[1:])
                    reshape = tf.reshape(pool2, [-1, dim])
                else:
                    dim = np.prod(pool1.get_shape().as_list()[1:])
                    reshape = tf.reshape(pool1, [-1, dim])

                pooled_outputs.append(reshape)

            with tf.name_scope("view_pooling"):
                x = tf.stack(pooled_outputs)  # 4 * N * dim
                x = tf.transpose(x, perm=[1, 2, 0])  # N * dim * 4
                reshape = tf.reshape(x, [-1, 4]) # (N * dim) * 4
                print reshape.get_shape().as_list()

                Weights = tf.Variable(tf.truncated_normal([4, 1], 0, 0.1), name="W")

                y = tf.matmul(reshape, Weights, name="view_pooling")
                y = tf.reshape(y, [-1, dim])
                print y.get_shape().as_list()
                print("DIM:!" + str(dim))

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(y, self.dropout_keep_prob, name="hidden_output_drop")
            print self.h_drop.get_shape().as_list()

        with tf.name_scope("fc1"):
            dim_ = dim / 2
            # dim_ = 100
            # W = tf.get_variable("W", [dim, dim_], initializer=tf.contrib.layers.xavier_initializer())
            W = tf.Variable(name="W", initial_value=tf.truncated_normal(shape=[dim, dim_], stddev=0.1))
            b = tf.Variable(tf.constant(0.1, shape=[dim_]), name="b")

            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.fc1 = tf.nn.relu(tf.matmul(self.h_drop, W) + b)
            self.fc_drop1 = tf.nn.dropout(self.fc1, self.dropout_keep_prob)

        # with tf.name_scope("fc2"):
        #     dim__ = dim_ / 2
        #     # dim_ = 100
        #     W = tf.Variable(name="W", initial_value=tf.truncated_normal(shape=[dim_, dim__], stddev=0.1))
        #     b = tf.Variable(tf.constant(0.1, shape=[dim__]), name="b")
        #
        #     l2_loss += tf.nn.l2_loss(W)
        #     l2_loss += tf.nn.l2_loss(b)
        #     self.fc2 = tf.nn.relu(tf.matmul(self.fc_drop1, W) + b)
        #     self.fc_drop2 = tf.nn.dropout(self.fc2, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            # W = tf.get_variable("W_output", [dim_, 2], initializer=tf.contrib.layers.xavier_initializer())
            W = tf.Variable(name="W_output", initial_value=tf.truncated_normal(shape=[dim_, 2], stddev=0.1))
            b = tf.Variable(tf.constant(0.1, shape=[2]), name="b")

            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.fc_drop1, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate Mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
