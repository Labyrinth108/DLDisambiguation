#!/usr/bin/env python
# encoding=utf-8

import datetime
import os
import time
import copy

import numpy as np
import tensorflow as tf

from util.util import write_evaluation_file
from DLDisambiguation.util.input_helpers import InputHelper
from MT_Dynamic_MultiGranModel import MT_Dynamic_MultiGranModel
from tensor import Tensor

# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of character embedding (default: 100)")
tf.flags.DEFINE_string("filter_sizes", "2,3", "Comma-separated filter sizes (default: '2,3')")
tf.flags.DEFINE_integer("num_filters", 16, "Number of filters per filter size (default: 64)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")

# Data Parameter
tf.flags.DEFINE_integer("max_sequence_len", 10, "max document length of input")
tf.flags.DEFINE_integer("max_sequence_len2", 10, "max document length of input")
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


def get_coocurrence(des_e_names, des_opr_map, x2_test, x4_test):
    res = []
    length = len(x2_test)

    for i in range(length):
        des = x2_test[i]
        opr = x4_test[i]
        N = 0.1

        if des_e_names.__contains__(des):
            index = des_e_names.index(des)
            des_opr = des_opr_map[index]
            if opr in des_opr.keys():
                N = des_opr[str(opr)]
        res.append(N)
    return np.expand_dims(np.asarray(res), axis=1)


def load_coocurrence_matrix(filename):
    t = open(filename)
    line = t.readline()
    des_e_names = []
    des_opr_map = []

    while line != "":
        des, oprs = line.split("\t")[0], line.split("\t")[1]
        des_e_names.append(des)

        oprs_num = oprs.split("_")[:-1]
        tmp = {}
        for opr in oprs_num:
            opr_name, num = opr.split(":")[0], int(opr.split(":")[1])
            tmp[opr_name] = num
        des_opr_map.append(copy.deepcopy(tmp))
        line = t.readline()
    return des_e_names, des_opr_map


def main():
    # Load data
    print("Loading data...")
    inputH = InputHelper()

    train_f = os.path.join(FLAGS.train_dir, 'data/exp0803/training_dynamic_data.txt')
    dev_f = os.path.join(FLAGS.train_dir, 'data/exp0803/validation_dynamic_data.txt')
    test_f = os.path.join(FLAGS.train_dir, 'data/exp0803/test_dynamic_data.txt')

    our_dir = "./Tensor_files/0803_dynamic/"
    # our_dir = "./Length" + str(FLAGS.max_sequence_len) + "/"
    x_train_tensor = np.load(our_dir + "train_des.npy")
    x_dev_tensor = np.load(our_dir + "dev_des.npy")
    x_test_tensor = np.load(our_dir + "test_des.npy")

    # our_dir = "./Length" + str(FLAGS.max_sequence_len2) + "/"
    x_train_tensor_o = np.load(our_dir + "train_opr.npy")
    x_dev_tensor_o = np.load(our_dir + "dev_opr.npy")
    x_test_tensor_o = np.load(our_dir + "test_opr.npy")

    x_train_indi_o = 1 - np.load(our_dir + "train_indi_opr.npy")
    x_dev_indi_o = 1 - np.load(our_dir + "dev_indi_opr.npy")
    x_test_indi_o = 1 - np.load(our_dir + "test_indi_opr.npy")

    sep = "\t"
    i1, x1_train, x2_train, x3_train, x4_train, y_train, y2_train = inputH.getTsvTestData_Mul_Labels_Dyna(train_f, sep,
                                                                                                          FLAGS.max_sequence_len)
    i2, x1_dev, x2_dev, x3_dev, x4_dev, y_dev, y2_dev = inputH.getTsvTestData_Mul_Labels_Dyna(dev_f, sep,
                                                                                              FLAGS.max_sequence_len)
    i3, x1_test, x2_test, x3_test, x4_test, y_test, y2_test = inputH.getTsvTestData_Mul_Labels_Dyna(test_f, sep,
                                                                                                    FLAGS.max_sequence_len)

    des_e_names, des_opr_map = load_coocurrence_matrix("result.txt")
    co_arr_test = get_coocurrence(des_e_names, des_opr_map, x2_test, x4_test)
    co_arr_train = get_coocurrence(des_e_names, des_opr_map, x2_train, x4_train)
    co_arr_val = get_coocurrence(des_e_names, des_opr_map, x2_dev, x4_dev)

    with tf.Graph().as_default():

        sess = tf.Session()
        with sess.as_default():
            cnn = MT_Dynamic_MultiGranModel(
                max_len1=FLAGS.max_sequence_len,
                max_len2=FLAGS.max_sequence_len2,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda,
            )

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

            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

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

            def train_step(x_batch, y_batch, x_batch2, y_batch2, indi, co_arr):
                gamma = [0.5 if i == 1 else 1.0 for i in indi]
                gamma = np.asarray(gamma)

                feed_dict = {
                    cnn.input_tensor: x_batch,
                    cnn.input_y_description: y_batch,
                    cnn.dropout_keep_prob: FLAGS.dropout_keep_prob,
                    cnn.input_tensor_o: x_batch2,
                    cnn.input_y_operation: y_batch2,
                    cnn.mask_opr: np.asarray(indi, dtype=float),
                    cnn.gamma: gamma,
                    cnn.matrix: co_arr
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

            def dev_step(x_dev, y_batch_dev, x_dev2, y_batch_dev2, indi, co_arr, writer=None):
                gamma = [0.5 if i == 1 else 1.0 for i in indi]
                gamma = np.asarray(gamma)
                feed_dict = {
                    cnn.input_tensor: x_dev,
                    cnn.input_y_description: y_batch_dev,
                    cnn.dropout_keep_prob: 1.0,
                    cnn.input_y_operation: y_batch_dev2,
                    cnn.input_tensor_o: x_dev2,
                    cnn.mask_opr: np.asarray(indi, dtype=float),
                    cnn.gamma: gamma,
                    cnn.matrix: co_arr
                }
                step, summaries, loss, accuracy1, accuracy2, pres1, pres2 = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy_d, cnn.accuracy_o, cnn.scores_d, cnn.scores_o],
                    feed_dict)
                if writer:
                    writer.add_summary(summaries, step)
                return loss, accuracy1, accuracy2

            def evaluate(x_dev, y_batch_dev, x_dev2, y_batch_dev2, indi, co_arr):
                gamma = [0.5 if i == 1 else 1.0 for i in indi]
                gamma = np.asarray(gamma)
                feed_dict = {
                    cnn.input_tensor: x_dev,
                    cnn.input_y_description: y_batch_dev,
                    cnn.dropout_keep_prob: 1.0,
                    cnn.input_y_operation: y_batch_dev2,
                    cnn.input_tensor_o: x_dev2,
                    cnn.mask_opr: np.asarray(indi, dtype=float),
                    cnn.gamma: gamma,
                    cnn.matrix: co_arr
                }

                step, summaries, loss, accuracy1, accuracy2, pres1, pres2 = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy_d, cnn.accuracy_o, cnn.scores_d, cnn.scores_o],
                    feed_dict)

                eval_file = open(out_dir + "/evaluation.txt", "w+")
                right_file = open(out_dir + "/right_cases.txt", "w+")
                right_file2 = open(out_dir + "/right_cases_operation.txt", "w+")
                wrong_file = open(out_dir + "/wrong_cases.txt", "w+")
                wrong_file2 = open(out_dir + "/wrong_cases_operation.txt", "w+")

                eval_file.write("Accu1: " + str(accuracy1) + "\n")
                eval_file.write("Accu2: " + str(accuracy2) + "\n")

                predictions1 = np.argmax(pres1, 1)
                predictions2 = np.argmax(pres2, 1)
                labels1 = np.argmax(y_batch_dev, 1)
                labels2 = np.argmax(y_batch_dev2, 1)

                def process(indi, tensor):
                    tmp = []
                    ll = len(indi)
                    for i in range(ll):
                        if indi[i] == 0:
                            tmp.append(tensor[i])
                    return np.asarray(tmp)

                write_evaluation_file(eval_file, right_file, wrong_file, labels1, predictions1, x1_test, x2_test)
                write_evaluation_file(eval_file, right_file2, wrong_file2, labels2, predictions2, x3_test, x4_test,
                                      indi)

                eval_file.write("Parameters:")
                for attr, value in sorted(FLAGS.__flags.items()):
                    eval_file.write("{}={}".format(attr.upper(), value) + "\n")

                return loss, accuracy1, accuracy2

            def dev_whole(x_dev, y_dev, x_dev2, y_dev2, indi, co_dev_arr, writer=None):
                batches_dev = inputH.batch_iter(list(zip(x_dev, y_dev, co_dev_arr)), FLAGS.batch_size, 1, shuffle=False)
                batches_dev2 = inputH.batch_iter(list(zip(x_dev2, y_dev2, indi)), FLAGS.batch_size, 1, shuffle=False)
                losses = []
                accuracies1 = []
                accuracies2 = []

                batches = zip(batches_dev, batches_dev2)

                for batches_dev, batches_dev2 in batches:
                    x_batch, y_batch, co_arr = zip(*batches_dev)
                    x_batch2, y_batch2, indi = zip(*batches_dev2)
                    loss, accuracy1, accuracy2 = dev_step(x_batch, y_batch, x_batch2, y_batch2, indi, co_arr, writer)
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
            batches = inputH.batch_iter(
                list(zip(x_train_tensor, y_train, x_train_tensor_o, y2_train, x_train_indi_o, co_arr_train)),
                FLAGS.batch_size, FLAGS.num_epochs)

            # Training loop. For each batch...
            dev_loss = []
            dev_loss2 = []
            # batch_d_o = zip(batches, batches2)
            for batch in batches:
                x_batch, y_batch, x_batch2, y_batch2, indi, co_arr = zip(*batch)

                train_step(x_batch, y_batch, x_batch2, y_batch2, indi, co_arr)
                current_step = tf.train.global_step(sess, global_step)

                if current_step % FLAGS.evaluate_every == 0:

                    print("\nEvaluation:")
                    loss, accuracy1, accuracy2 = dev_whole(x_dev_tensor, y_dev, x_dev_tensor_o, y2_dev, x_dev_indi_o,
                                                           co_arr_val, writer=dev_summary_writer)

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

            loss, accuracy1, accuracy2 = evaluate(x_test_tensor, y_test, x_test_tensor_o, y2_test, x_test_indi_o,
                                                  co_arr_test)
            print(loss)
            print(accuracy1)
            print(accuracy2)


if __name__ == '__main__':
    main()
