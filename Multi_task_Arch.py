#!/usr/bin/env python
# encoding=utf-8

import datetime
import os
import time

import numpy as np
import tensorflow as tf

from DLDisambiguation.util.input_helpers import InputHelper
from MultiTask_MultiGranModel import MultiTask_MultiGranModel
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


def generate_Tensor(mention, entity, max_len, task_n):
    tensor = Tensor(mention, entity, len(mention), max_len, task_n).get_tensor()
    tensor = tensor.transpose((0, 2, 3, 1))
    return tensor


def prepara_tensor_y(inputH, file, max_len):
    x1_train, x2_train, x3_train, x4_train, y_train, y2_train, \
    x1_dev, x2_dev, x3_dev, x4_dev, y_dev, y2_dev = inputH.splitMulti_DataSets(file, 30, max_len)

    x_train_tensor = generate_Tensor(x1_train, x2_train, max_len, 1)
    x_dev_tensor = generate_Tensor(x1_dev, x2_dev, max_len, 1)

    x_train_tensor_o = generate_Tensor(x3_train, x4_train, max_len, 2)
    x_dev_tensor_o = generate_Tensor(x3_dev, x4_dev, max_len, 2)

    return x_train_tensor, y_train, x_dev_tensor, y_dev, x_train_tensor_o, y2_train, x_dev_tensor_o, y2_dev


def main():
    # Load data
    print("Loading data...")
    inputH = InputHelper()

    data_f = os.path.join(FLAGS.train_dir, 'data/toy_mul')
    # data_f = os.path.join(FLAGS.train_dir, 'data/train_data_0720')

    x_train_tensor, y_train, x_dev_tensor, y_dev, x_train_tensor_o, y_train_o, x_dev_tensor_o, y_dev_o \
        = prepara_tensor_y(inputH, data_f, FLAGS.max_sequence_len)

    with tf.Graph().as_default():

        sess = tf.Session()
        with sess.as_default():
            cnn = MultiTask_MultiGranModel(
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
            out_dir = os.path.abspath(os.path.join(FLAGS.train_dir, "runs", "multitask" + timestamp))
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            print("Writing to {}\n".format(out_dir))
            checkpoint_prefix = os.path.join(out_dir, "model")

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary1 = tf.summary.scalar("accuracy1", cnn.accuracy_d)
            acc_summary2 = tf.summary.scalar("accuracy2", cnn.accuracy_o)

            # Train Summaries
            train_summary_op = tf.summary.merge_all()
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary1, acc_summary2])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Initialize all variables
            sess.run(tf.initialize_all_variables())

            def train_step(x_batch, y_batch, x_batch2, y_batch2):
                feed_dict = {
                    cnn.input_tensor: x_batch,
                    cnn.input_y_description: y_batch,
                    cnn.dropout_keep_prob: FLAGS.dropout_keep_prob,
                    cnn.input_tensor_o: x_batch2,
                    cnn.input_y_operation: y_batch2,
                }
                _, step, summaries, loss, accuracy1, accuracy2 = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy_d, cnn.accuracy_o],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                if step % 10 == 0:
                    print(
                        "{}: step {}, loss {:g}, acc1 {:g}, acc2 {:g}".format(time_str, step, loss, accuracy1,
                                                                              accuracy2))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_dev, y_batch_dev, x_dev2, y_batch_dev2, writer=None):
                feed_dict = {
                    cnn.input_tensor: x_dev,
                    cnn.input_y_description: y_batch_dev,
                    cnn.dropout_keep_prob: 1.0,
                    cnn.input_y_operation: y_batch_dev2,
                    cnn.input_tensor_o: x_dev2,
                }
                step, summaries, loss, accuracy1, accuracy2, pres1, pres2 = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy_d, cnn.accuracy_o, cnn.scores_d, cnn.scores_o],
                    feed_dict)
                if writer:
                    writer.add_summary(summaries, step)
                return loss, accuracy1, accuracy2

            def dev_whole(x_dev, y_dev, x_dev2, y_dev2, writer=None):
                batches_dev = inputH.batch_iter(list(zip(x_dev, y_dev)), FLAGS.batch_size, 1, shuffle=False)
                batches_dev2 = inputH.batch_iter(list(zip(x_dev2, y_dev2)), FLAGS.batch_size, 1, shuffle=False)
                losses = []
                accuracies1 = []
                accuracies2 = []

                batches = zip(batches_dev, batches_dev2)

                for batches_dev, batches_dev2 in batches:
                    x_batch, y_batch = zip(*batches_dev)
                    x_batch2, y_batch2 = zip(*batches_dev2)
                    loss, accuracy1, accuracy2 = dev_step(x_batch, y_batch, x_batch2, y_batch2, writer)
                    losses.append(loss)
                    accuracies1.append(accuracy1)
                    accuracies2.append(accuracy2)
                return np.mean(np.array(losses)), np.mean(np.array(accuracies1)), np.mean(np.array(accuracies2))

            def overfit(dev_loss):
                n = len(dev_loss)
                if n < 5:
                    return False
                for i in xrange(n - 4, n):
                    if dev_loss[i] > dev_loss[i - 1]:
                        return False
                return True

            # Generate batches
            batches = inputH.batch_iter(list(zip(x_train_tensor, y_train, x_train_tensor_o, y_train_o)),
                                        FLAGS.batch_size, FLAGS.num_epochs)
            # batches2 = inputH.batch_iter(list(zip(x_train_tensor_o, y_train_o)), FLAGS.batch_size, FLAGS.num_epochs)

            # Training loop. For each batch...
            dev_loss = []
            dev_loss2 = []
            # batch_d_o = zip(batches, batches2)
            for batch in batches:
                x_batch, y_batch, x_batch2, y_batch2 = zip(*batch)
                # x_batch2, y_batch2 = zip(*batch2)

                train_step(x_batch, y_batch, x_batch2, y_batch2)
                current_step = tf.train.global_step(sess, global_step)

                if current_step % FLAGS.evaluate_every == 0:

                    print("\nEvaluation:")
                    loss, accuracy1, accuracy2 = dev_whole(x_dev_tensor, y_dev, x_dev_tensor_o, y_dev_o,
                                                           writer=dev_summary_writer)

                    time_str = datetime.datetime.now().isoformat()
                    print("{}: dev-aver, loss {:g}, acc {:g}, acc2 {:g}".format(time_str, loss, accuracy1, accuracy2))
                    dev_loss.append(accuracy1)
                    dev_loss2.append(accuracy2)

                    print("\nRecently accuracy:")
                    print dev_loss[-10:]
                    print dev_loss2[-10:]

                    if overfit(dev_loss):
                        print 'Overfit!! in task1'
                        break
                    if overfit(dev_loss2):
                        print 'Overfit!! in task2'
                        break
                    print("")

                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))


if __name__ == '__main__':
    main()
