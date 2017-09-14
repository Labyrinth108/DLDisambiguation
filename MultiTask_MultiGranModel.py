#!/usr/bin/env python
# encoding=utf-8
import numpy as np
import tensorflow as tf


class MultiTask_MultiGranModel(object):
    def _conv(self, name, in_, ksize, reuse=False):
        num_filters = ksize[3]

        with tf.variable_scope(name, reuse=reuse) as scope:
            # different CNN for different views
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

    def __init__(self, max_len1, max_len2, filter_sizes, pool_sizes, filter_sizes2, pool_sizes2, num_filters,
                 l2_reg_lambda=0.0, constraint_lambda=0.0, alpha=0.5, type_CNN=2, view_num=0, view_nums=[]):
        channel_num = 4

        # Placeholders for input, output and dropout
        self.input_tensor = tf.placeholder(tf.float32, [None, max_len1, max_len1, channel_num], name="input_tensor_description")
        self.input_tensor_o = tf.placeholder(tf.float32, [None, max_len2, max_len2, channel_num], name="input_tensor_operation")

        self.input_y_description = tf.placeholder(tf.float32, [None, 2], name="input_y_description")
        self.input_y_operation = tf.placeholder(tf.float32, [None, 2], name="input_y_operation")

        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.matrix = tf.placeholder(tf.float32, [None, 1], name="cooccurence")
        self.constraint_lambda = constraint_lambda

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

        if type_CNN == 2: # multi-view
            with tf.name_scope("CNN_Description"):
                view_c_num = 0
                for i in range(channel_num):
                    # set reuse True for i > 0, for weight-sharing
                    reuse_f = (i != 0)
                    with tf.variable_scope("CNN_Description", reuse=reuse_f):
                        if len(view_nums) != 0:
                            if len(view_nums) <= view_c_num or view_nums[view_c_num] != i:
                                continue
                            else:
                                view_c_num += 1
                        print("AHAA" + str(i) + "\n")
                        view = tf.gather(input_tensor, i)  # N x W x H x C

                        filter_shape1 = [filter_sizes[0], filter_sizes[0], 1, num_filters]
                        filter_shape2 = [filter_sizes[1], filter_sizes[1], num_filters, num_filters * 2]
                        p_size1 = [1, pool_sizes[0], pool_sizes[0], 1]
                        p_size2 = [1, pool_sizes[1], pool_sizes[1], 1]

                        conv1 = self._conv('conv1', view, filter_shape1, reuse=reuse_f)
                        pool1 = self._maxpool('pool1', conv1, ksize=p_size1, strides=[1, 1, 1, 1])

                        conv2 = self._conv('conv2', pool1, filter_shape2, reuse=reuse_f)
                        pool2 = self._maxpool('pool2', conv2, ksize=p_size2, strides=[1, 1, 1, 1])

                        dim1 = np.prod(pool2.get_shape().as_list()[1:])
                        reshape = tf.reshape(pool2, [-1, dim1])

                        pooled_outputs.append(reshape)

            with tf.name_scope("CNN_Operation"):
                view_c_num = 0
                for i in range(channel_num):
                    # set reuse True for i > 0, for weight-sharing
                    reuse_f = (i != 0)

                    with tf.variable_scope("CNN_Operation", reuse=reuse_f):
                        if len(view_nums) != 0:
                            if len(view_nums) <= view_c_num or view_nums[view_c_num] != i:
                                continue
                            else:
                                view_c_num += 1
                        print("AHAA" + str(i) + "\n")
                        view = tf.gather(input_tensor_operation, i)  # N x W x H x C

                        filter_shape1 = [filter_sizes2[0], filter_sizes2[0], 1, num_filters / 2]
                        filter_shape2 = [filter_sizes2[1], filter_sizes2[1], num_filters / 2, num_filters]
                        p_size1 = [1, pool_sizes2[0], pool_sizes2[0], 1]
                        p_size2 = [1, pool_sizes2[1], pool_sizes2[1], 1]

                        conv1 = self._conv('conv1', view, filter_shape1, reuse=reuse_f)
                        pool1 = self._maxpool('pool1', conv1, ksize=p_size1, strides=[1, 1, 1, 1])

                        conv2 = self._conv('conv2', pool1, filter_shape2, reuse=reuse_f)
                        pool2 = self._maxpool('pool2', conv2, ksize=p_size2, strides=[1, 1, 1, 1])

                        dim2 = np.prod(pool2.get_shape().as_list()[1:])
                        reshape = tf.reshape(pool2, [-1, dim2])

                        pooled_outputs_operation.append(reshape)


            view_num_len = len(pooled_outputs)
            print("LEN:" + str(view_num_len))
            with tf.name_scope("Descriptipn_view_pooling"):
                x = tf.stack(pooled_outputs)  # 4 * N * 7744
                x = tf.transpose(x, perm=[1, 2, 0])  # N * 7744 * 4

                reshape = tf.reshape(x, [-1, view_num_len])
                print reshape.get_shape().as_list()

                Weights = tf.Variable(tf.random_uniform([view_num_len, 1], 0.0, 1.0), name="W")

                y_d = tf.matmul(reshape, Weights, name="view_pooling")
                y_d = tf.reshape(y_d, [-1, dim1])
                print y_d.get_shape().as_list()

            with tf.name_scope("Operation_view_pooling"):
                x = tf.stack(pooled_outputs_operation)  # 4 * N * 7744
                x = tf.transpose(x, perm=[1, 2, 0])  # N * 7744 * 4
                reshape = tf.reshape(x, [-1, view_num_len])
                print reshape.get_shape().as_list()

                Weights = tf.Variable(tf.random_uniform([view_num_len, 1], 0.0, 1.0), name="W")

                y_o = tf.matmul(reshape, Weights, name="view_pooling")
                y_o = tf.reshape(y_o, [-1, dim2])
                print y_o.get_shape().as_list()

        elif type_CNN == 3: # single view
            with tf.name_scope("CNN_Description"):
                view = tf.gather(input_tensor, view_num)  # N x W x H x C
                filter_shape1 = [filter_sizes[0], filter_sizes[0], 1, num_filters]
                filter_shape2 = [filter_sizes[1], filter_sizes[1], num_filters, num_filters * 2]
                p_size1 = [1, pool_sizes[0], pool_sizes[0], 1]
                p_size2 = [1, pool_sizes[1], pool_sizes[1], 1]

                conv1 = self._conv("conv1", view, filter_shape1)
                pool1 = self._maxpool('pool1', conv1, ksize=p_size1, strides=[1, 1, 1, 1])
                conv2 = self._conv('conv2', pool1, filter_shape2)
                pool2 = self._maxpool('pool2', conv2, ksize=p_size2, strides=[1, 1, 1, 1])

                dim1 = np.prod(pool2.get_shape().as_list()[1:])
                y_d = tf.reshape(pool2, [-1, dim1])

            with tf.name_scope("CNN_Operation"):
                view = tf.gather(input_tensor_operation, view_num)  # N x W x H x C
                filter_shape1 = [filter_sizes2[0], filter_sizes2[0], 1, num_filters / 2]
                filter_shape2 = [filter_sizes2[1], filter_sizes2[1], num_filters / 2, num_filters]
                p_size1 = [1, pool_sizes2[0], pool_sizes2[0], 1]
                p_size2 = [1, pool_sizes2[1], pool_sizes2[1], 1]

                conv1 = self._conv('conv1', view, filter_shape1)
                pool1 = self._maxpool('pool1', conv1, ksize=p_size1, strides=[1, 1, 1, 1])

                conv2 = self._conv('conv2', pool1, filter_shape2)
                pool2 = self._maxpool('pool2', conv2, ksize=p_size2, strides=[1, 1, 1, 1])

                dim2 = np.prod(pool2.get_shape().as_list()[1:])
                y_o = tf.reshape(pool2, [-1, dim2])

        else: # single CNN
            with tf.name_scope("CNN_Description"):
                filter_shape1 = [filter_sizes[0], filter_sizes[0], 4, num_filters]
                filter_shape2 = [filter_sizes[1], filter_sizes[1], num_filters, num_filters * 2]
                p_size1 = [1, pool_sizes[0], pool_sizes[0], 1]
                p_size2 = [1, pool_sizes[1], pool_sizes[1], 1]

                conv1 = self._conv("conv1", self.input_tensor, filter_shape1)
                pool1 = self._maxpool('pool1', conv1, ksize=p_size1, strides=[1, 1, 1, 1])
                conv2 = self._conv('conv2', pool1, filter_shape2)
                pool2 = self._maxpool('pool2', conv2, ksize=p_size2, strides=[1, 1, 1, 1])

                dim1 = np.prod(pool2.get_shape().as_list()[1:])
                y_d = tf.reshape(pool2, [-1, dim1])

            with tf.name_scope("CNN_Operation"):
                filter_shape1 = [filter_sizes2[0], filter_sizes2[0], 4, num_filters / 2]
                filter_shape2 = [filter_sizes2[1], filter_sizes2[1], num_filters / 2, num_filters]
                p_size1 = [1, pool_sizes2[0], pool_sizes2[0], 1]
                p_size2 = [1, pool_sizes2[1], pool_sizes2[1], 1]

                conv1 = self._conv('conv1', self.input_tensor_o, filter_shape1)
                pool1 = self._maxpool('pool1', conv1, ksize=p_size1, strides=[1, 1, 1, 1])

                conv2 = self._conv('conv2', pool1, filter_shape2)
                pool2 = self._maxpool('pool2', conv2, ksize=p_size2, strides=[1, 1, 1, 1])

                dim2 = np.prod(pool2.get_shape().as_list()[1:])
                y_o = tf.reshape(pool2, [-1, dim2])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop_d = tf.nn.dropout(y_d, self.dropout_keep_prob, name="hidden_output_description_drop")
            self.h_drop_o = tf.nn.dropout(y_o, self.dropout_keep_prob, name="hidden_output_operation_drop")
            print self.h_drop_d.get_shape().as_list()
            print self.h_drop_o.get_shape().as_list()

        with tf.name_scope("FC"):
            dim = min(int(dim1 / 2), int(dim2 / 2))
            print("FC DIM:" + str(dim) + "\n")
            W1 = tf.Variable(name="W1", initial_value=tf.truncated_normal(shape=[dim1, dim], stddev=0.1))
            b1 = tf.Variable(tf.constant(0.1, shape=[dim]), name="b1")

            self.fc_d = tf.nn.relu(tf.matmul(self.h_drop_d, W1) + b1)
            self.fc_drop_d = tf.nn.dropout(self.fc_d, self.dropout_keep_prob)

            W2 = tf.Variable(name="W2", initial_value=tf.truncated_normal(shape=[dim2, dim], stddev=0.1))
            b2 = tf.Variable(tf.constant(0.1, shape=[dim]), name="b2")

            self.fc_o = tf.nn.relu(tf.matmul(self.h_drop_o, W2) + b2)
            self.fc_drop_o = tf.nn.dropout(self.fc_o, self.dropout_keep_prob)

        # Share Layer Construction
        with tf.name_scope("Multitask"):

            self.shared_layer = tf.add(alpha * self.fc_drop_d, (1 - alpha) * self.fc_drop_o, name="Shared_layer")
            # self.shared_layer = tf.div(tf.add(self.h_drop_d, self.h_drop_o), 2, name="Shared_layer")
            print self.shared_layer.get_shape().as_list()

            W1 = tf.get_variable(name="tt1_W", shape=[dim],
                                 initializer=tf.truncated_normal_initializer(stddev=0.1))
            W2 = tf.get_variable(name="st1_W", shape=[dim],
                                 initializer=tf.truncated_normal_initializer(stddev=0.1))
            W3 = tf.get_variable(name="st2_W", shape=[dim],
                                 initializer=tf.truncated_normal_initializer(stddev=0.1))
            W4 = tf.get_variable(name="tt2_W", shape=[dim],
                                 initializer=tf.truncated_normal_initializer(stddev=0.1))

            self.task1_r = tf.add(tf.multiply(self.shared_layer, W2), tf.multiply(self.fc_drop_d, W1),
                                  name="description_r")
            self.task2_r = tf.add(tf.multiply(self.shared_layer, W3), tf.multiply(self.fc_drop_o, W4),
                                  name="operation_r")
            print self.task1_r.get_shape().as_list()

        with tf.name_scope("FC2"):
            W1 = tf.Variable(name="W1", initial_value=tf.truncated_normal(shape=[dim, dim / 2], stddev=0.1))
            b1 = tf.Variable(tf.constant(0.1, shape=[dim / 2]), name="b1")

            self.task1_representation = tf.nn.relu(tf.matmul(self.task1_r, W1) + b1)
            self.task1_representation = tf.nn.dropout(self.task1_representation, self.dropout_keep_prob)

            W2 = tf.Variable(name="W2", initial_value=tf.truncated_normal(shape=[dim, dim / 2], stddev=0.1))
            b2 = tf.Variable(tf.constant(0.1, shape=[dim / 2]), name="b2")

            self.task2_representation = tf.nn.relu(tf.matmul(self.task2_r, W2) + b2)
            self.task2_representation = tf.nn.dropout(self.task2_representation, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W_d = tf.get_variable(name="W_d", shape=[dim / 2, 2],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
            b_d = tf.Variable(tf.constant(0.1, shape=[2]), name="b_d")

            l2_loss_d += tf.nn.l2_loss(W_d)
            l2_loss_d += tf.nn.l2_loss(b_d)

            W_o = tf.get_variable(name="W_o", shape=[dim / 2, 2],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
            b_o = tf.Variable(tf.constant(0.1, shape=[2]), name="b_o")

            l2_loss_operation += tf.nn.l2_loss(W_o)
            l2_loss_operation += tf.nn.l2_loss(b_o)

            self.scores_d = tf.nn.xw_plus_b(self.task1_representation, W_d, b_d, name="scores1")
            self.scores_o = tf.nn.xw_plus_b(self.task2_representation, W_o, b_o, name="scores2")

            self.relation_d = tf.nn.softmax(self.scores_d, name="relation1")
            self.relation_o = tf.nn.softmax(self.scores_o, name="relation2")

            self.predictions_d = tf.argmax(self.scores_d, 1, name="predictions1")
            self.predictions_o = tf.argmax(self.scores_o, 1, name="predictions2")

        # Calculate Mean cross-entropy loss
        with tf.name_scope("loss"):
            losses1 = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores_d, labels=self.input_y_description)
            losses2 = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores_o, labels=self.input_y_operation)

            gap = tf.reduce_sum(tf.square(self.relation_d - self.relation_o), axis=1, keep_dims=True)
            constraints = tf.multiply(self.matrix, gap)
            self.constraints = tf.identity(tf.reduce_mean(constraints), name="constraints")

            self.loss = tf.reduce_mean(losses1) + tf.reduce_mean(losses2) + l2_reg_lambda * (
                l2_loss_d + l2_loss_operation) + self.constraint_lambda * tf.reduce_mean(constraints)
            self.loss = tf.identity(self.loss, name="loss")

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions_d = tf.equal(self.predictions_d, tf.argmax(self.input_y_description, 1))
            correct_predictions_o = tf.equal(self.predictions_o, tf.argmax(self.input_y_operation, 1))
            self.accuracy_d = tf.reduce_mean(tf.cast(correct_predictions_d, "float"), name="accuracy_d")
            self.accuracy_o = tf.reduce_mean(tf.cast(correct_predictions_o, "float"), name="accuracy_o")
