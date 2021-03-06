#!/usr/bin/env python
# encoding=utf-8

import datetime
import os
import time

import numpy as np
import tensorflow as tf

from util.util import write_evaluation_file
from util.input_helpers import InputHelper
from MultiGran_Model import MultiGranModel
from tensor import Tensor

# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of character embedding (default: 100)")
tf.flags.DEFINE_string("filter_sizes", "3, 3, 3, 3", "Comma-separated filter sizes (default: '2,3')")
tf.flags.DEFINE_string("pool_sizes", "2, 2, 2, 2", "Comma-separated filter sizes (default: '2,3')")
tf.flags.DEFINE_integer("num_filters", 16, "Number of filters per filter size (default: 64)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.001, "L2 regularizaion lambda (default: 0.0)")

# Data Parameter
tf.flags.DEFINE_integer("max_sequence_len", 20, "max document length of input")
tf.flags.DEFINE_integer("most_words", 300000, "Most number of words in vocab (default: 300000)")

# Training parameters
tf.flags.DEFINE_integer("seed", 123, "Random seed (default: 123)")
tf.flags.DEFINE_string("train_dir", "./", "Training dir root")
tf.flags.DEFINE_integer("batch_size", 128, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 50, "Number of training epochs")
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
    print("Loading data...")

    type_CNN = 2
    run_type = "Single CNN" if type_CNN == 1 else "Multiview 4CNN"

    early_stop_num = 11
    task_num = 2
    name = "des" if task_num == 1 else "opr"
    d_type = "Single_task1" if task_num == 1 else "Single_task2"

    inputH = InputHelper()
    max_document_length = FLAGS.max_sequence_len
    load_from_npy = True

    sep = "\t"
    time_gen = "0823"
    data_file = os.path.join(FLAGS.train_dir, "data/train_data_" + time_gen + "_" + name + ".txt")
    data_file_test = os.path.join(FLAGS.train_dir, "data/test_data_" + time_gen + "_" + name + ".txt")
    # data_file_val = os.path.join(FLAGS.train_dir, "data/validation_data_" + time_gen + "_" + name + ".txt")
    data_file_val = data_file_test

    # data_file = os.path.join(FLAGS.train_dir, "data/training_dynamic_data_" + name + ".txt")
    # data_file_test = os.path.join(FLAGS.train_dir, "data/test_dynamic_data_" + name + ".txt")
    # data_file_val = os.path.join(FLAGS.train_dir, "data/validation_dynamic_data_" + name + ".txt")

    x_train_mention, x_train_entity, y_train = inputH.getTsvTestData(data_file, sep, max_document_length, y_value=False)
    x_dev_mention, x_dev_entity, y_dev = inputH.getTsvTestData(data_file_val, sep, max_document_length, y_value=False)
    x_test_mention, x_test_entity, y_test = inputH.getTsvTestData(data_file_test, sep, max_document_length,
                                                                  y_value=False)

    if load_from_npy == False:
        # Constructing Tensor for train, dev, and test
        men_arr = np.concatenate((x_train_mention, x_dev_mention, x_test_mention))
        entity_arr = np.concatenate((x_train_entity, x_dev_entity, x_test_entity))

        lstm_dir = "Description1501174251" if task_num == 1 else "Operation1501209225"
        bilstm_dir = os.path.join("./Sentence_Modeling/runs", lstm_dir)

        tensor = Tensor(men_arr, entity_arr, len(men_arr), FLAGS.max_sequence_len, task_num, bilstm_dir).get_tensor()
        tensor = tensor.transpose((0, 2, 3, 1))
        g1 = len(x_train_mention)
        g2 = len(np.concatenate((x_train_mention, x_dev_mention)))
        x_train_tensor, x_dev_tensor, x_test_tensor = tensor[:g1], tensor[g1:g2], tensor[g2:]

    else:
        mydir = "./Tensor_files/0823/Length" + str(FLAGS.max_sequence_len) + "/"

        x_train_tensor = np.load(mydir + "train_" + name + ".npy")
        # x_dev_tensor = np.load(mydir + "dev_" + name + ".npy")
        x_test_tensor = np.load(mydir + "test_" + name + ".npy")
        x_dev_tensor = x_test_tensor

        # indi_train = np.load(mydir + "train_indi_" + name + ".npy")
        # indi_val = np.load(mydir + "dev_indi_" + name + ".npy")
        # indi_test = np.load(mydir + "test_indi_" + name + ".npy")

        # def process(indi, tensor):
        #     tmp = []
        #     ll = len(indi)
        #     for i in range(ll):
        #         if indi[i] == 0:
        #             tmp.append(tensor[i])
        #     return np.asarray(tmp)
        #
        # x_train_tensor = process(indi_train, x_train_tensor)
        # x_dev_tensor = process(indi_val, x_dev_tensor)
        # x_test_tensor = process(indi_test, x_test_tensor)

    with tf.Graph().as_default():

        sess = tf.Session()
        with sess.as_default():
            cnn = MultiGranModel(
                max_len=FLAGS.max_sequence_len,
                pool_sizes=list(map(int, FLAGS.pool_sizes.split(","))),
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda,
                type_CNN=type_CNN)  # Define Training procedure
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
            out_dir = os.path.abspath(os.path.join(FLAGS.train_dir, "runs", "NewExp", d_type + timestamp))
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            print("Writing to {}\n".format(out_dir))

            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

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

            def evaluate(x_dev, y_batch_dev, m):
                feed_dict = {
                    m.input_tensor: x_dev,
                    m.input_y: y_batch_dev,
                    m.dropout_keep_prob: 1.0
                }
                step, loss, accuracy, pres = sess.run([global_step, m.loss, m.accuracy, m.scores],
                                                      feed_dict)

                eval_file = open(out_dir + "/evaluation.txt", "w+")
                right_file = open(out_dir + "/right_cases.txt", "w+")
                wrong_file = open(out_dir + "/wrong_cases.txt", "w+")

                eval_file.write("Accu: " + str(accuracy) + "\n")
                predictions = np.argmax(pres, 1)
                labels = np.argmax(y_batch_dev, 1)

                write_evaluation_file(eval_file, right_file, wrong_file, labels, predictions,
                                      x_test_mention,
                                      x_test_entity)

                eval_file.write("Parameters:")
                for attr, value in sorted(FLAGS.__flags.items()):
                    eval_file.write("{}={}".format(attr.upper(), value) + "\n")
                fs1 = [str(i) for i in fs]
                ps1 = [str(i) for i in ps]
                eval_file.write("Conv" + " ".join(fs1) + "\n")
                eval_file.write("Pool_sizes" + " ".join(ps1) + "\n")

                print(loss)
                print(accuracy)
                print(pres)
                print(y_batch_dev)
                return loss, accuracy

            def dev_whole(x_dev, y_dev, writer=None):
                # batches_dev = inputH.batch_iter(list(zip(x_dev, y_dev)), FLAGS.batch_size, 1,
                #                                 shuffle=False)
                # losses = []
                # accuracies = []
                #
                # for idx, batch_dev in enumerate(batches_dev):
                #     x_batch, y_batch = zip(*batch_dev)
                #     loss, accurary = dev_step(x_batch, y_batch, writer)
                #     losses.append(loss)
                #     accuracies.append(accurary)
                # return np.mean(np.array(losses)), np.mean(np.array(accuracies))

                l, accu = dev_step(x_dev, y_dev, writer)
                return l, accu

            def overfit(dev_loss, accu):

                n = len(dev_loss)
                if n < early_stop_num:
                    return False

                # mean_acc = np.mean(dev_loss[-15:])
                # if mean_acc < accu:
                #     return False
                for i in xrange(n - early_stop_num, n):
                    if dev_loss[i] < accu:
                        return False
                print(dev_loss)
                print(accu)
                return True

            # Generate batches
            batches = inputH.batch_iter(list(zip(x_train_tensor, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

            data_num = len(y_train)
            num_batches_per_epoch = int(data_num / FLAGS.batch_size)
            # num_batches_per_epoch_2 = int(num_batches_per_epoch / 2)
            print num_batches_per_epoch

            # Training loop. For each batch...
            dev_loss = []
            optimum_accu = 0

            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)

                if current_step % num_batches_per_epoch == 0:
                    print("\nEvaluation:")
                    loss, accuracy = dev_whole(x_dev_tensor, y_dev, writer=dev_summary_writer)
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: dev-aver, loss {:g}, acc {:g}".format(time_str, loss, accuracy))
                    dev_loss.append(accuracy)

                    print("\nRecently accuracy:")
                    print dev_loss[-10:]
                    if overfit(dev_loss, accuracy):
                        print 'Overfit!!'
                        print(current_step)
                        print(current_step / num_batches_per_epoch)
                        break
                    print("")

                    if accuracy > optimum_accu:
                        optimum_accu = accuracy
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        print("Saved model checkpoint to {}\n".format(path))
                    print("Optimum_accu: " + str(optimum_accu))

            print("")
            print("Optimum_accu: " + str(optimum_accu))

        # evaluate the result with the best model
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        checkpoint_file = ckpt.model_checkpoint_path
        graph = tf.Graph()

        with graph.as_default():
            sess = tf.Session()
            with sess.as_default():
                # Load the saved meta graph and restore variables
                saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
                sess.run(tf.initialize_all_variables())
                saver.restore(sess, checkpoint_file)

                input_t = graph.get_operation_by_name("input_tensor").outputs[0]
                input_y = graph.get_operation_by_name("input_y").outputs[0]
                prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

                prediction = graph.get_operation_by_name("output/predictions").outputs[0]
                accu = graph.get_operation_by_name("accuracy/accuracy").outputs[0]
                pres, accuracy = sess.run([prediction, accu], {input_t: x_test_tensor, input_y: y_test, prob: 1})

                labels = np.argmax(y_test, 1)
                eval_file = open(out_dir + "/evaluation.txt", "w+")
                right_file = open(out_dir + "/right_cases.txt", "w+")
                wrong_file = open(out_dir + "/wrong_cases.txt", "w+")

                eval_file.write("Dataset: " + data_file + "\n")
                eval_file.write(run_type + "\n")
                eval_file.write("Stopped at " + str(current_step / num_batches_per_epoch) + "\n")
                eval_file.write("Accu: " + str(accuracy) + "\n")
                write_evaluation_file(eval_file, right_file, wrong_file, labels, pres, x_test_mention,
                                      x_test_entity)

                eval_file.write("Parameters:")
                for attr, value in sorted(FLAGS.__flags.items()):
                    eval_file.write("{}={}".format(attr.upper(), value) + "\n")

                print("loss:" + str(loss))
                print("accuracy:" + str(accuracy))


if __name__ == '__main__':
    main()
