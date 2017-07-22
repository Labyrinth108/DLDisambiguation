#!/usr/bin/env python
# encoding=utf-8
import numpy as np
import tensorflow as tf


class MultiTask_MultiGranModel(object):
    def _conv(self, name, in_, ksize, reuse=False):
        num_filters = ksize[3]

        with tf.variable_scope(name, reuse=reuse) as scope:
            # different CNN for different views
            # W = tf.Variable(tf.truncated_normal(ksize, stddev=0.1), name="W")
            # biases = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")

            # same CNN for different views
            # W = tf.get_variable("weights", ksize, initializer=tf.contrib.layers.xavier_initializer())
            W = tf.get_variable("weights", ksize, initializer=tf.truncated_normal_initializer(stddev=0.1))
            biases = tf.get_variable("biases", [num_filters], initializer=tf.constant_initializer(0.1))

            conv = tf.nn.conv2d(in_, W, strides=[1, 1, 1, 1], padding="VALID")
            h = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope.name)

        return h

    def _maxpool(self, name, in_, ksize, strides):
        pool = tf.nn.max_pool(in_, ksize=ksize, strides=strides, padding='VALID', name=name)
        print name, pool.get_shape().as_list()
        return pool

    def __init__(self, max_len, filter_sizes, num_filters, l2_reg_lambda=0.0):
        channel_num = 4

        # Placeholders for input, output and dropout
        self.input_tensor = tf.placeholder(tf.float32, [None, max_len, max_len, 4], name="input_tensor_description")
        self.input_tensor_o = tf.placeholder(tf.float32, [None, max_len, max_len, 4], name="input_tensor_operation")

        self.input_y_description = tf.placeholder(tf.float32, [None, 2], name="input_y_description")
        self.input_y_operation = tf.placeholder(tf.float32, [None, 2], name="input_y_operation")

        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss_d = tf.constant(0.0)
        l2_loss_operation = tf.constant(0.0)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        pooled_outputs_operation = []

        input_tensor = tf.expand_dims(self.input_tensor, 4)  # N x W x H x V  => N x W x H x V x C
        input_tensor = tf.transpose(input_tensor,
                                    perm=[3, 0, 1, 2, 4])  # N x W x H x V x C =>  V x N x W x H x C

        input_tensor_operation = tf.expand_dims(self.input_tensor_o, 4)  # N x W x H x V  => N x W x H x V x C
        input_tensor_operation = tf.transpose(input_tensor_operation,
                                              perm=[3, 0, 1, 2, 4])  # N x W x H x V x C =>  V x N x W x H x C

        with tf.name_scope("CNN_Description"):
            for i in range(channel_num):
                # set reuse True for i > 0, for weight-sharing
                reuse_f = (i != 0)
                with tf.variable_scope("CNN_Description", reuse=reuse_f):
                    view = tf.gather(input_tensor, i)  # N x W x H x C

                    filter_shape1 = [filter_sizes[0], filter_sizes[0], 1, num_filters / 2]
                    filter_shape2 = [filter_sizes[1], filter_sizes[1], num_filters / 2, num_filters]
                    p_size1 = [1, 3, 3, 1]
                    p_size2 = [1, 5, 5, 1]

                    conv1 = self._conv('conv1', view, filter_shape1, reuse=reuse_f)
                    pool1 = self._maxpool('pool1', conv1, ksize=p_size1, strides=[1, 1, 1, 1])

                    conv2 = self._conv('conv2', pool1, filter_shape2, reuse=reuse_f)
                    pool2 = self._maxpool('pool2', conv2, ksize=p_size2, strides=[1, 1, 1, 1])

                    dim = np.prod(pool2.get_shape().as_list()[1:])
                    reshape = tf.reshape(pool2, [-1, dim])

                    pooled_outputs.append(reshape)

        with tf.name_scope("CNN_Operation"):
            for i in range(channel_num):
                # set reuse True for i > 0, for weight-sharing
                reuse_f = (i != 0)

                with tf.variable_scope("CNN_Operation", reuse=reuse_f):
                    view = tf.gather(input_tensor_operation, i)  # N x W x H x C

                    filter_shape1 = [filter_sizes[0], filter_sizes[0], 1, num_filters / 2]
                    filter_shape2 = [filter_sizes[1], filter_sizes[1], num_filters / 2, num_filters]
                    p_size1 = [1, 3, 3, 1]
                    p_size2 = [1, 5, 5, 1]

                    conv1 = self._conv('conv1', view, filter_shape1, reuse=reuse_f)
                    pool1 = self._maxpool('pool1', conv1, ksize=p_size1, strides=[1, 1, 1, 1])

                    conv2 = self._conv('conv2', pool1, filter_shape2, reuse=reuse_f)
                    pool2 = self._maxpool('pool2', conv2, ksize=p_size2, strides=[1, 1, 1, 1])

                    dim = np.prod(pool2.get_shape().as_list()[1:])
                    reshape = tf.reshape(pool2, [-1, dim])

                    pooled_outputs_operation.append(reshape)

        with tf.name_scope("Descriptipn_view_pooling"):
            x = tf.stack(pooled_outputs)  # 4 * N * 7744
            x = tf.transpose(x, perm=[1, 2, 0])  # N * 7744 * 4
            reshape = tf.reshape(x, [-1, 4])
            print reshape.get_shape().as_list()

            Weights = tf.Variable(tf.random_uniform([4, 1], 0.0, 1.0), name="W")

            y_d = tf.matmul(reshape, Weights, name="view_pooling")
            y_d = tf.reshape(y_d, [-1, dim])
            print y_d.get_shape().as_list()

        with tf.name_scope("Operation_view_pooling"):
            x = tf.stack(pooled_outputs_operation)  # 4 * N * 7744
            x = tf.transpose(x, perm=[1, 2, 0])  # N * 7744 * 4
            reshape = tf.reshape(x, [-1, 4])
            print reshape.get_shape().as_list()

            Weights = tf.Variable(tf.random_uniform([4, 1], 0.0, 1.0), name="W")

            y_o = tf.matmul(reshape, Weights, name="view_pooling")
            y_o = tf.reshape(y_o, [-1, dim])
            print y_o.get_shape().as_list()

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop_d = tf.nn.dropout(y_d, self.dropout_keep_prob, name="hidden_output_description_drop")
            self.h_drop_o = tf.nn.dropout(y_o, self.dropout_keep_prob, name="hidden_output_operation_drop")
            print self.h_drop_d.get_shape().as_list()
            print self.h_drop_o.get_shape().as_list()

        # Share Layer Construction
        with tf.name_scope("Multitask"):

            self.shared_layer = tf.div(tf.add(self.h_drop_d, self.h_drop_o), 2, name="Shared_layer")
            print self.shared_layer.get_shape().as_list()

            W1 = tf.Variable(name="tt1_W", initial_value=tf.random_normal([dim], stddev=0.1))
            W2 = tf.Variable(name="st1_W", initial_value=tf.random_normal([dim], stddev=0.1))
            W3 = tf.Variable(name="st2_W", initial_value=tf.random_normal([dim], stddev=0.1))
            W4 = tf.Variable(name="tt2_W", initial_value=tf.random_normal([dim], stddev=0.1))

            self.task1_r = tf.add(tf.multiply(self.shared_layer, W2), tf.multiply(self.h_drop_d, W1),
                                  name="description_r")
            self.task2_r = tf.add(tf.multiply(self.shared_layer, W3), tf.multiply(self.h_drop_o, W4),
                                  name="operation_r")
            print self.task1_r.get_shape().as_list()

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W_d = tf.Variable(name="W_output_d", initial_value=tf.random_normal([dim, 2], stddev=0.1))
            b_d = tf.Variable(tf.constant(0.1, shape=[2]), name="b_d")

            l2_loss_d += tf.nn.l2_loss(W_d)
            l2_loss_d += tf.nn.l2_loss(b_d)

            W_o = tf.Variable(name="W_output_o", initial_value=tf.random_normal([dim, 2], stddev=0.1))
            b_o = tf.Variable(tf.constant(0.1, shape=[2]), name="b_o")

            l2_loss_operation += tf.nn.l2_loss(W_o)
            l2_loss_operation += tf.nn.l2_loss(b_o)

            self.scores_d = tf.nn.xw_plus_b(self.task1_r, W_d, b_d, name="scores1")
            self.scores_o = tf.nn.xw_plus_b(self.task2_r, W_o, b_o, name="scores2")
            self.predictions_d = tf.argmax(self.scores_d, 1, name="predictions1")
            self.predictions_o = tf.argmax(self.scores_o, 1, name="predictions2")

        # Calculate Mean cross-entropy loss
        with tf.name_scope("loss"):
            losses1 = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores_d, labels=self.input_y_description)
            losses2 = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores_o, labels=self.input_y_operation)
            self.loss = tf.reduce_mean(losses1) + tf.reduce_mean(losses2) + l2_reg_lambda * (
                l2_loss_d + l2_loss_operation)

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions_d = tf.equal(self.predictions_d, tf.argmax(self.input_y_description, 1))
            correct_predictions_o = tf.equal(self.predictions_o, tf.argmax(self.input_y_operation, 1))
            self.accuracy_d = tf.reduce_mean(tf.cast(correct_predictions_d, "float"), name="accuracy_d")
            self.accuracy_o = tf.reduce_mean(tf.cast(correct_predictions_o, "float"), name="accuracy_o")