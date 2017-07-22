#!/usr/bin/env python
# encoding=utf-8

import datetime
import os
import time

import numpy as np
import tensorflow as tf

from DLDisambiguation.util.input_helpers import InputHelper
# from DeepModel import CNNModel
from MultiGran_Model import MultiGranModel
from tensor import Tensor

# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of character embedding (default: 100)")
tf.flags.DEFINE_string("filter_sizes", "2,3", "Comma-separated filter sizes (default: '2,3')")
tf.flags.DEFINE_integer("num_filters", 64, "Number of filters per filter size (default: 64)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")

# Data Parameter
tf.flags.DEFINE_integer("max_sequence_len", 20, "max document length of input")
tf.flags.DEFINE_integer("most_words", 300000, "Most number of words in vocab (default: 300000)")

# Training parameters
tf.flags.DEFINE_integer("seed", 123, "Random seed (default: 123)")
tf.flags.DEFINE_string("train_dir", "./", "Training dir root")
tf.flags.DEFINE_integer("batch_size", 128, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 5, "Number of training epochs")
tf.flags.DEFINE_float("eval_split", 0.1, "Use how much data for evaluating (default: 0.1)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


def main():
    # Load data
    print("Loading data...")
    inputH = InputHelper()

    # x_train_mention, x_train_entity, y_train = inputH.getTsvTestData(
    #     os.path.join(FLAGS.train_dir, 'data/train0.txt'))
    # x_dev_mention, x_dev_entity, y_dev = inputH.getTsvTestData(
    #     os.path.join(FLAGS.train_dir, 'data/test0.txt'))

    # data_file = os.path.join(FLAGS.train_dir, 'data/toy_data.txt') # 测试用
    # data_file = os.path.join(FLAGS.train_dir, 'data/testing_data.txt') # description -- 自己生成的--小
    # data_file = os.path.join(FLAGS.train_dir, 'data/training_data.txt')  # description-自己生成的--大

    # data_file = os.path.join(FLAGS.train_dir, 'data/train_data_0720_operation.txt') # operation
    data_file = os.path.join(FLAGS.train_dir, 'data/new_training_data_.txt')  # description

    task_num = 1
    x_train_mention, x_train_entity, y_train, x_dev_mention, x_dev_entity, y_dev = inputH.splitDataSets(
        data_file, 30, FLAGS.max_sequence_len)  # 30---percent_test--30%

    x_train_tensor = Tensor(x_train_mention, x_train_entity, len(x_train_entity), FLAGS.max_sequence_len,
                            task_num).get_tensor()
    x_train_tensor = x_train_tensor.transpose((0, 2, 3, 1))

    x_dev_tensor = Tensor(x_dev_mention, x_dev_entity, len(x_dev_mention), FLAGS.max_sequence_len,
                          task_num).get_tensor()
    x_dev_tensor = x_dev_tensor.transpose((0, 2, 3, 1))

    with tf.Graph().as_default():

        sess = tf.Session()
        with sess.as_default():
            cnn = MultiGranModel(
                max_len=FLAGS.max_sequence_len,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
            saver = tf.train.Saver(tf.all_variables(), max_to_keep=20)

            # Keep track of gradient values and sparsity (optional)
            for g, v in grads_and_vars:
                if g is not None:
                    tf.summary.histogram("grad_hist/{}".format(v.name), g)
                    tf.summary.scalar("grad_sparsity/{}".format(v.name), tf.nn.zero_fraction(g))
                    tf.summary.histogram(v.name, v)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(FLAGS.train_dir, "runs", timestamp))
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            print("Writing to {}\n".format(out_dir))
            checkpoint_prefix = os.path.join(out_dir, "model")

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_merged = tf.summary.merge_all()
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Initialize all variables
            sess.run(tf.initialize_all_variables())

            def train_step(x_batch, y_batch):
                feed_dict = {
                    cnn.input_tensor: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_merged, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                if step % 10 == 0:
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_dev, y_batch_dev, writer=None):
                feed_dict = {
                    cnn.input_tensor: x_dev,
                    cnn.input_y: y_batch_dev,
                    cnn.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy, pres = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy, cnn.scores],
                    feed_dict)
                if writer:
                    writer.add_summary(summaries, step)
                return loss, accuracy

            def dev_whole(x_dev, y_dev, writer=None):
                batches_dev = inputH.batch_iter(list(zip(x_dev, y_dev)), FLAGS.batch_size, 1, shuffle=False)
                losses = []
                accuracies = []

                for idx, batch_dev in enumerate(batches_dev):
                    x_batch, y_batch = zip(*batch_dev)
                    loss, accurary = dev_step(x_batch, y_batch, writer)
                    losses.append(loss)
                    accuracies.append(accurary)
                return np.mean(np.array(losses)), np.mean(np.array(accuracies))

            def overfit(dev_loss):
                n = len(dev_loss)
                if n < 5:
                    return False
                for i in xrange(n - 4, n):
                    if dev_loss[i] > dev_loss[i - 1]:
                        return False
                return True

            # Generate batches
            batches = inputH.batch_iter(list(zip(x_train_tensor, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

            # Training loop. For each batch...
            dev_loss = []
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)

                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    loss, accuracy = dev_whole(x_dev_tensor, y_dev, writer=dev_summary_writer)
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: dev-aver, loss {:g}, acc {:g}".format(time_str, loss, accuracy))
                    dev_loss.append(accuracy)
                    print("\nRecently accuracy:")
                    print dev_loss[-10:]
                    if overfit(dev_loss):
                        print 'Overfit!!'
                        break
                    print("")

                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))


if __name__ == '__main__':
    main()
