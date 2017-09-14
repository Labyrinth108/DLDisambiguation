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
from MultiTask_MultiGranModel import MultiTask_MultiGranModel
from tensor import Tensor

# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of character embedding (default: 100)")
tf.flags.DEFINE_string("filter_sizes", "4,3", "Comma-separated filter sizes (default: '2,3')")
tf.flags.DEFINE_string("filter_sizes2", "3,3", "Comma-separated filter sizes (default: '2,3')")
tf.flags.DEFINE_string("pool_sizes", "2,2", "Comma-separated filter sizes (default: '2,3')")
tf.flags.DEFINE_string("pool_sizes2", "3,3", "Comma-separated filter sizes (default: '2,3')")
tf.flags.DEFINE_string("view_nums", "0,1,2,3", "view combination")

tf.flags.DEFINE_integer("num_filters", 16, "Number of filters per filter size (default: 64)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 100, "L2 regularizaion lambda (default: 0.0)")
tf.flags.DEFINE_float("con_lambda", 0.1, "constraint regularizaion lambda (default: 0.0)")
tf.flags.DEFINE_float("alpha", 0.5, "parameter for shared layer")
tf.flags.DEFINE_integer("type_CNN", 2, "type of CNN")
tf.flags.DEFINE_integer("view_num", 3, "type of CNN")
tf.flags.DEFINE_boolean("early_stop", True, "whether early stopping is used")
tf.flags.DEFINE_integer("early_stop_num", 11, "number of epoch in early stopping")

# Data Parameter
tf.flags.DEFINE_integer("max_sequence_len", 10, "max document length of input")
tf.flags.DEFINE_integer("max_sequence_len2", 20, "max document length of input")
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


def generate_Tensor(mention, entity, mention2, entity2, mention3, entity3, max_len, task_n):
    lstm_dir = "Description1501058401" if task_n == 1 else "Operation1501000120"
    bilstm_dir = os.path.join("./Sentence_Modeling/runs", lstm_dir)

    tensor = Tensor(mention + mention2 + mention3, entity + entity2 + entity3, len(mention + mention2 + mention3),
                    max_len, task_n, bilstm_dir).get_tensor()
    tensor = tensor.transpose((0, 2, 3, 1))

    g1 = len(mention)
    g2 = len(mention + mention2)
    return tensor[:g1], tensor[g1:g2], tensor[g2:]


def prepara_tensor_y(inputH, training_path, dev_path, test_path, max_len):
    sep = "\t"
    x1_train, x2_train, x3_train, x4_train, y_train, y2_train = inputH.getTsvTestData_Mul(training_path, sep, max_len)
    x1_dev, x2_dev, x3_dev, x4_dev, y_dev, y2_dev = inputH.getTsvTestData_Mul(dev_path, sep, max_len)
    x1_test, x2_test, x3_test, x4_test, y_test, y2_test = inputH.getTsvTestData_Mul(test_path, sep, max_len)

    x_train_tensor, x_dev_tensor, x_test_tensor = generate_Tensor(x1_train, x2_train, x1_dev, x2_dev, x1_test, x2_test,
                                                                  max_len, 1)

    x_train_tensor_o, x_dev_tensor_o, x_test_tensor_o = generate_Tensor(x3_train, x4_train, x3_dev, x4_dev, x3_test,
                                                                        x4_test, max_len, 2)

    np.save("train_des", x_train_tensor)
    np.save("dev_des", x_dev_tensor)
    np.save("test_des", x_test_tensor)

    np.save("train_opr", x_train_tensor_o)
    np.save("dev_opr", x_dev_tensor_o)
    np.save("test_opr", x_test_tensor_o)

    return x_train_tensor, y_train, x_dev_tensor, y_dev, x_test_tensor, y_test, \
           x_train_tensor_o, y2_train, x_dev_tensor_o, y2_dev, x_test_tensor_o, y2_test


def get_coocurrence(des_e_names, des_opr_map, x2_test, x4_test, N):
    res = []
    length = len(x2_test)

    for i in range(length):
        des = x2_test[i]
        opr = x4_test[i]

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

    date_f = "0823"
    train_f = "./data/exp" + date_f + "/data_augment_train.txt"
    test_f = "./data/exp" + date_f + "/data_augment_test.txt"
    dev_f = test_f

    our_dir = "./Tensor_files/" + date_f + "/Length" + str(FLAGS.max_sequence_len) + "/"
    x_train_tensor = np.load(our_dir + "train_des.npy")
    # x_dev_tensor = np.load(our_dir + "dev_des.npy")
    x_test_tensor = np.load(our_dir + "test_des.npy")
    x_dev_tensor = x_test_tensor

    our_dir = "./Tensor_files/" + date_f + "/Length" + str(FLAGS.max_sequence_len2) + "/"
    x_train_tensor_o = np.load(our_dir + "train_opr.npy")
    # x_dev_tensor_o = np.load(our_dir + "dev_opr.npy")
    x_test_tensor_o = np.load(our_dir + "test_opr.npy")
    x_dev_tensor_o = x_test_tensor_o

    def normalize(a):
        amin, amax = a.min(), a.max()  # 求最大最小值
        a = (a - amin) / (amax - amin)  # (矩阵元素-最小值)/(最大值-最小值)
        return a

    def normalize_tensor(t):
        t[:, :, :, 0] = normalize(t[:, :, :, 0])
        t[:, :, :, 1] = normalize(t[:, :, :, 1])
        t[:, :, :, 2] = normalize(t[:, :, :, 2])
        t[:, :, :, 3] = normalize(t[:, :, :, 3])
        return t

    x_test_tensor[:, :, :, 3] = normalize(x_test_tensor[:, :, :, 3])
    x_train_tensor[:, :, :, 3] = normalize(x_train_tensor[:, :, :, 3])
    x_test_tensor_o[:, :, :, 3] = normalize(x_test_tensor_o[:, :, :, 3])
    x_train_tensor_o[:, :, :, 3] = normalize(x_train_tensor_o[:, :, :, 3])
    # x_test_tensor = normalize_tensor(x_test_tensor)
    # x_test_tensor_o = normalize_tensor(x_test_tensor_o)
    # x_train_tensor = normalize_tensor(x_train_tensor)
    # x_train_tensor_o = normalize_tensor(x_train_tensor_o)

    sep = "\t"
    x1_train, x2_train, x3_train, x4_train, y_train, y2_train = inputH.getTsvTestData_Mul_Labels(train_f, sep,
                                                                                                 FLAGS.max_sequence_len)
    # x1_dev, x2_dev, x3_dev, x4_dev, y_dev, y2_dev = inputH.getTsvTestData_Mul_Labels(dev_f, sep, FLAGS.max_sequence_len)
    x1_test, x2_test, x3_test, x4_test, y_test, y2_test = inputH.getTsvTestData_Mul_Labels(test_f, sep,
                                                                                           FLAGS.max_sequence_len)
    x1_dev, x2_dev, x3_dev, x4_dev, y_dev, y2_dev = x1_test, x2_test, x3_test, x4_test, y_test, y2_test

    des_e_names, des_opr_map = load_coocurrence_matrix("coorrence_file.txt")
    N_default = 0.01
    co_arr_test = get_coocurrence(des_e_names, des_opr_map, x2_test, x4_test, N_default)
    co_arr_train = get_coocurrence(des_e_names, des_opr_map, x2_train, x4_train, N_default)
    # co_arr_val = get_coocurrence(des_e_names, des_opr_map, x2_dev, x4_dev)
    co_arr_val = co_arr_test

    with tf.Graph().as_default():

        sess = tf.Session()
        with sess.as_default():
            cnn = MultiTask_MultiGranModel(
                max_len1=FLAGS.max_sequence_len,
                max_len2=FLAGS.max_sequence_len2,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                filter_sizes2=list(map(int, FLAGS.filter_sizes2.split(","))),
                pool_sizes=list(map(int, FLAGS.pool_sizes.split(","))),
                pool_sizes2=list(map(int, FLAGS.pool_sizes2.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda,
                constraint_lambda=FLAGS.con_lambda,
                alpha=FLAGS.alpha,
                type_CNN=FLAGS.type_CNN,
                view_num=FLAGS.view_num,
                view_nums=list(map(int, FLAGS.view_nums.split(",")))
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
            out_dir = os.path.abspath(os.path.join(FLAGS.train_dir, "runs", "alpha_E", "multitask" + timestamp))
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            print("Writing to {}\n".format(out_dir))

            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            constraint_summary = tf.summary.scalar("constraints", cnn.constraints)
            acc_summary1 = tf.summary.scalar("accuracy1", cnn.accuracy_d)
            acc_summary2 = tf.summary.scalar("accuracy2", cnn.accuracy_o)

            # Train Summaries
            train_summary_op = tf.summary.merge_all()
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary1, acc_summary2, constraint_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Initialize all variables
            sess.run(tf.initialize_all_variables())

            def train_step(x_batch, y_batch, x_batch2, y_batch2, co_arr):
                feed_dict = {
                    cnn.input_tensor: x_batch,
                    cnn.input_y_description: y_batch,
                    cnn.dropout_keep_prob: FLAGS.dropout_keep_prob,
                    cnn.input_tensor_o: x_batch2,
                    cnn.input_y_operation: y_batch2,
                    cnn.matrix: co_arr,
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
                return accuracy1, accuracy2, loss

            def dev_step(x_dev, y_batch_dev, x_dev2, y_batch_dev2, co_arr, writer=None):
                feed_dict = {
                    cnn.input_tensor: x_dev,
                    cnn.input_y_description: y_batch_dev,
                    cnn.dropout_keep_prob: 1.0,
                    cnn.input_y_operation: y_batch_dev2,
                    cnn.input_tensor_o: x_dev2,
                    cnn.matrix: co_arr
                }
                step, summaries, loss, accuracy1, accuracy2, pres1, pres2 = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy_d, cnn.accuracy_o, cnn.scores_d, cnn.scores_o],
                    feed_dict)
                if writer:
                    writer.add_summary(summaries, step)
                return loss, accuracy1, accuracy2

            def evaluate(x_dev, y_batch_dev, x_dev2, y_batch_dev2, co_arr):
                feed_dict = {
                    cnn.input_tensor: x_dev,
                    cnn.input_y_description: y_batch_dev,
                    cnn.dropout_keep_prob: 1.0,
                    cnn.input_y_operation: y_batch_dev2,
                    cnn.input_tensor_o: x_dev2,
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
                write_evaluation_file(eval_file, right_file, wrong_file, labels1, predictions1, x1_test, x2_test)
                write_evaluation_file(eval_file, right_file2, wrong_file2, labels2, predictions2, x3_test, x4_test)

                eval_file.write("Parameters:")
                for attr, value in sorted(FLAGS.__flags.items()):
                    eval_file.write("{}={}".format(attr.upper(), value) + "\n")

                return loss, accuracy1, accuracy2

            def dev_whole(x_dev, y_dev, x_dev2, y_dev2, co_arr, writer=None):
                batches_dev = inputH.batch_iter(list(zip(x_dev, y_dev, co_arr)), FLAGS.batch_size, 1, shuffle=False)
                batches_dev2 = inputH.batch_iter(list(zip(x_dev2, y_dev2)), FLAGS.batch_size, 1, shuffle=False)
                losses = []
                accuracies1 = []
                accuracies2 = []

                batches = zip(batches_dev, batches_dev2)

                for batches_dev, batches_dev2 in batches:
                    x_batch, y_batch, co_arr_ = zip(*batches_dev)
                    x_batch2, y_batch2 = zip(*batches_dev2)
                    loss, accuracy1, accuracy2 = dev_step(x_batch, y_batch, x_batch2, y_batch2, co_arr_)
                    losses.append(loss)
                    accuracies1.append(accuracy1)
                    accuracies2.append(accuracy2)

                return np.mean(np.array(losses)), np.mean(np.array(accuracies1)), np.mean(np.array(accuracies2))

            def overfit(dev_loss, accu):
                num = FLAGS.early_stop_num
                n = len(dev_loss)
                if n < num:
                    return False

                for i in xrange(n - num, n - 1):
                    if dev_loss[i] < accu:
                        return False
                print(dev_loss)
                print(accu)
                return True

            # Generate batches
            batches = inputH.batch_iter(list(zip(x_train_tensor, y_train, x_train_tensor_o, y2_train, co_arr_train)),
                                        FLAGS.batch_size, FLAGS.num_epochs)

            # Training loop. For each batch...
            dev_loss = []
            train_loss = []

            train_accu = []
            train_accu2 = []
            dev_accu = []
            dev_accu2 = []
            # batch_d_o = zip(batches, batches2)
            optimum_accu1 = 0
            optimum_accu2 = 0
            data_num = len(y_train)
            num_batches_per_epoch = int(data_num / FLAGS.batch_size)
            # t = num_batches_per_epoch / 2
            optimum_loss = 1000

            for batch in batches:
                x_batch, y_batch, x_batch2, y_batch2, co_arr_batch = zip(*batch)

                acc1, acc2, loss_train = train_step(x_batch, y_batch, x_batch2, y_batch2, co_arr_batch)
                train_accu.append(acc1)
                train_accu2.append(acc2)
                train_loss.append(loss_train)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % num_batches_per_epoch == 0:

                    print("\nEvaluation:")
                    loss, accuracy1, accuracy2 = dev_whole(x_dev_tensor, y_dev, x_dev_tensor_o, y2_dev, co_arr_val,
                                                           writer=dev_summary_writer)

                    summary = tf.Summary()

                    summary.value.add(tag="Accuracy_Dev", simple_value=accuracy1)
                    summary.value.add(tag="Accuracy2_Dev", simple_value=accuracy2)
                    summary.value.add(tag="Loss_Dev", simple_value=loss)
                    dev_summary_writer.add_summary(summary, current_step)

                    time_str = datetime.datetime.now().isoformat()
                    print("{}: dev-aver, loss {:g}, acc {:g}, acc2 {:g}".format(time_str, loss, accuracy1, accuracy2))
                    dev_accu.append(accuracy1)
                    dev_accu2.append(accuracy2)
                    dev_loss.append(loss)
                    print("\nRecently accuracy:")
                    print dev_accu[-10:]
                    print dev_accu2[-10:]

                    # if loss < optimum_loss:
                    #     optimum_loss = loss
                    #     stop_early = 0
                    #     optimum_accu1 = accuracy1
                    #     optimum_accu2 = accuracy2
                    #     path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    #     print("Saved model checkpoint to {}\n".format(path))
                    # else:
                    #     stop_early += 1
                    #     if stop_early == 10:
                    #         break
                    if FLAGS.early_stop:
                        if overfit(dev_accu, accuracy1) or overfit(dev_accu2, accuracy2):
                            print 'Overfit!!'
                            print(current_step)
                            print(current_step / num_batches_per_epoch)
                            break
                        print("")

                    if accuracy1 > optimum_accu1 and accuracy2 > optimum_accu2:
                        optimum_accu1 = accuracy1
                        optimum_accu2 = accuracy2
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        print("Saved model checkpoint to {}\n".format(path))

                    print("Optimum_accu1: " + str(optimum_accu1))
                    print("Optimum_accu2: " + str(optimum_accu2))

    print("Optimum_accu1: " + str(optimum_accu1))
    print("Optimum_accu2: " + str(optimum_accu2))

    import matplotlib.pyplot as plt
    # def plot_plots(y1, y2, name_task, type_eval):
    #     x1 = np.arange(len(y1))
    #     x2 = np.arange(len(y2))
    #     p1, = plt.plot(x1, y1, 'b', label="Validation")
    #     p2, = plt.plot(x2, y2, 'r', label="Train")
    #     plt.legend(handles=[p1, p2], numpoints=1)  # make legend
    #     plt.title(name_task + "_" + type_eval)
    #     plt.savefig(os.path.join(out_dir, name_task + "_" + type_eval + ".png"))
    #
    # plot_plots(dev_accu, train_accu, "Disease", "Accu")
    # plot_plots(dev_accu2, train_accu2, "Operation", "Accu")
    # plot_plots(dev_loss, train_loss, "MTL", "Loss")

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

            cooccur = graph.get_operation_by_name("cooccurence").outputs[0]
            input_t1 = graph.get_operation_by_name("input_tensor_description").outputs[0]
            input_t2 = graph.get_operation_by_name("input_tensor_operation").outputs[0]
            input_y1 = graph.get_operation_by_name("input_y_description").outputs[0]
            input_y2 = graph.get_operation_by_name("input_y_operation").outputs[0]
            prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            loss_opr = graph.get_operation_by_name("loss/loss").outputs[0]
            prediction = graph.get_operation_by_name("output/predictions1").outputs[0]
            prediction2 = graph.get_operation_by_name("output/predictions2").outputs[0]
            accu = graph.get_operation_by_name("accuracy/accuracy_d").outputs[0]
            accu2 = graph.get_operation_by_name("accuracy/accuracy_o").outputs[0]

            loss, pres1, pres2, accuracy1, accuracy2 = sess.run(
                [loss_opr, prediction, prediction2, accu, accu2],
                {input_t1: x_test_tensor, input_y1: y_test, cooccur: co_arr_test,
                 input_t2: x_test_tensor_o, input_y2: y2_test, prob: 1})

            eval_file = open(out_dir + "/evaluation.txt", "w+")
            right_file = open(out_dir + "/right_cases.txt", "w+")
            wrong_file = open(out_dir + "/wrong_cases.txt", "w+")
            right_file2 = open(out_dir + "/right_cases_opr.txt", "w+")
            wrong_file2 = open(out_dir + "/wrong_cases_opr.txt", "w+")

            eval_file.write("Accu1: " + str(accuracy1) + "\n")
            eval_file.write("Accu2: " + str(accuracy2) + "\n")
            # eval_file.write("Stopped at: " + str(int(current_step / num_batches_per_epoch)) + "\n")
            eval_file.write("Default: " + str(N_default) + "\n")

            labels1 = np.argmax(y_test, 1)
            labels2 = np.argmax(y2_test, 1)
            write_evaluation_file(eval_file, right_file, wrong_file, labels1, pres1, x1_test, x2_test)
            write_evaluation_file(eval_file, right_file2, wrong_file2, labels2, pres2, x3_test, x4_test)

            eval_file.write("Parameters:")
            for attr, value in sorted(FLAGS.__flags.items()):
                eval_file.write("{}={}".format(attr.upper(), value) + "\n")

            print("loss:" + str(loss))
            print("accuracy1:" + str(accuracy1))
            print("accuracy2:" + str(accuracy2))


if __name__ == '__main__':
    main()
