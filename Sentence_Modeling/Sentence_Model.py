import tensorflow as tf
from Siamese_network import SiameseLSTM
from DLDisambiguation.util.input_helpers import InputHelper
from DLDisambiguation.util.util import getEmbedding
from DLDisambiguation.util.preprocess import MyVocabularyProcessor
import time
import numpy as np
import datetime
import os

from DLDisambiguation.util.util import write_evaluation_file

# Parameters
# ==================================================
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of character embedding")
tf.flags.DEFINE_integer("hidden_units", 15, "unit numbers of hidden vectors in Bi-LSTM")
tf.flags.DEFINE_float("l2_reg_lambda", 0.01, "L2 regularization lambda (default: 0.0)")

tf.flags.DEFINE_string("train_dir", "../", "training dir")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 128, "Batch Size")
tf.flags.DEFINE_integer("num_epochs", 50, "Number of training epochs (default: 50)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


class SentenceModel:
    def getEmbeddingMatrix(self, embedding_dir, processer):
        vocab, vocab_size, embedding_dim, embedding = getEmbedding(embedding_dir)

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

    def __init__(self):
        # the max length of description/operation segment, padding if shorter, and ignore the pair if longer
        max_document_length = 20
        inpH = InputHelper()
        y_is_value = True  # flag to indicate that y is value(0 / 1) or array[0,1] / [1, 0]
        # train_set, dev_set, vocab_processor, sum_no_of_batches = inpH.getDataSets_File(FLAGS.training_files, "\t",
        #                                                                           max_document_length,
        #                                                                           10,  # 10---percent_dev
        #                                                                           FLAGS.batch_size, y_value=y_is_value)

        # test_x1, test_x2, test_y = inpH.getTestDataSet(FLAGS.test_file, "\t\t", vocab_processor, max_document_length, y_is_value)

        task_num = 2
        d_type = "Description" if task_num == 1 else "Operation"
        embedding_dir = "../data/word2vec/character_model.txt" if task_num == 1 \
            else "../data/operation/character_model.txt"
        name = "des" if task_num == 1 else "opr"

        time_gen = "0823"
        data_file = os.path.join(FLAGS.train_dir, "data/train_data_" + time_gen + "_" + name + ".txt")
        data_file_test = os.path.join(FLAGS.train_dir, "data/test_data_" + time_gen + "_" + name + ".txt")
        # data_file_val = os.path.join(FLAGS.train_dir, "data/validation_data_" + time_gen + "_" + name + ".txt")

        sep = "\t"
        train_x1, train_x2, train_y = inpH.getTsvTestData(data_file, sep, max_document_length, y_is_value)
        test_x1, test_x2, test_y = inpH.getTsvTestData(data_file_test, sep, max_document_length, y_is_value)
        # dev_x1, dev_x2, dev_y = inpH.getTsvTestData(data_file_val, sep, max_document_length, y_is_value)
        dev_x1, dev_x2, dev_y = test_x1, test_x2, test_y
        sum_no_of_batches = len(train_y) // FLAGS.batch_size

        vocab_processor = MyVocabularyProcessor(max_document_length, min_frequency=0)
        vocab_processor.fit_transform(np.concatenate((train_x1, train_x2, dev_x1, dev_x2)))
        # vocab_processor.fit_transform(np.concatenate((train_x1, train_x2, test_x1, test_x2, dev_x1, dev_x2)))
        print("Length of loaded vocabulary ={}".format(len(vocab_processor.vocabulary_)))

        train_set = inpH.get_data(vocab_processor, train_x1, train_x2, train_y, max_document_length)
        dev_set = inpH.get_data(vocab_processor, dev_x1, dev_x2, dev_y, max_document_length)
        test_set = inpH.get_data(vocab_processor, test_x1, test_x2, test_y, max_document_length)

        # load in word2vec model
        embedding_matrix = self.getEmbeddingMatrix(embedding_dir, vocab_processor)

        sess = tf.Session()
        with sess.as_default():
            siameseModel = SiameseLSTM(sequence_length=max_document_length,
                                       vocab_processer=vocab_processor,
                                       embedding_size=FLAGS.embedding_dim,
                                       hidden_unit_size=FLAGS.hidden_units,
                                       l2_reg_lambda=FLAGS.l2_reg_lambda,
                                       batch_size=FLAGS.batch_size,
                                       embedding_arr=embedding_matrix)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            print("initialized siameseModel object")

            grads_and_vars = optimizer.compute_gradients(siameseModel.loss)
            tr_op_set = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
            print("defined training_ops")

            # Keep track of variables, gradient values and sparsity
            for g, v in grads_and_vars:
                if g is not None:
                    tf.summary.histogram("grad_hist/{}".format(v.name), g)
                    tf.summary.histogram("grad_sparsity/{}".format(v.name), tf.nn.zero_fraction(g))
                    tf.summary.histogram(v.name, v)
            print("defined gradient summaries")

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "Exp" + time_gen, "runs", d_type + timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", siameseModel.loss)
            acc_summary = tf.summary.scalar("accuracy", siameseModel.accuracy)

            # Train Summaries
            train_summary_merged = tf.summary.merge_all()
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.all_variables(), max_to_keep=100)

            # Write vocabulary
            vocab_processor.save(os.path.join(checkpoint_dir, "vocab"))

            # Initialize all variables
            sess.run(tf.initialize_all_variables())
            print("init all variables")

            graph_def = tf.get_default_graph().as_graph_def()
            graphpb_txt = str(graph_def)
            with open(os.path.join(checkpoint_dir, "graphpb.txt"), 'w') as f:
                f.write(graphpb_txt)

            def train_step(x1_batch, x2_batch, x1_batch_m, x2_batch_m, y_batch):

                feed_dict = {
                    siameseModel.input_x1: x1_batch,
                    siameseModel.input_x2: x2_batch,
                    siameseModel.mask_x1: x1_batch_m,
                    siameseModel.mask_x2: x2_batch_m,
                    siameseModel.input_y: y_batch,
                }

                _, step, summaries, loss, accuracy, dist = sess.run(
                    [tr_op_set, global_step, train_summary_merged, siameseModel.loss, siameseModel.accuracy,
                     siameseModel.distance], feed_dict)

                time_str = datetime.datetime.now().isoformat()
                d = np.copy(dist)
                d[d >= 0.5] = 999.0
                d[d < 0.5] = 1
                d[d > 1.0] = 0
                accuracy_t = np.mean(y_batch == d)
                print(
                    "TRAIN {}: step {}, loss {:g}, acc {:g}, acc_t {:g}".format(time_str, step, loss, accuracy, accuracy_t))
                print(y_batch)
                print(dist)
                print(d)
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x1_batch, x2_batch, x1_batch_m, x2_batch_m, y_batch):

                feed_dict = {
                    siameseModel.input_x1: x1_batch,
                    siameseModel.input_x2: x2_batch,
                    siameseModel.mask_x1: x1_batch_m,
                    siameseModel.mask_x2: x2_batch_m,
                    siameseModel.input_y: y_batch,
                }

                step, summaries, loss, accuracy, dist = sess.run(
                    [global_step, dev_summary_op, siameseModel.loss, siameseModel.accuracy, siameseModel.distance],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                d = np.copy(dist)
                d[d >= 0.5] = 999.0
                d[d < 0.5] = 1
                d[d > 1.0] = 0
                accuracy_t = np.mean(y_batch == d)
                print("DEV {}: step {}, loss {:g}, acc {:g}, acc_t {:g}".format(time_str, step, loss, accuracy, accuracy_t))
                print(y_batch)
                print(dist)
                print(d)
                dev_summary_writer.add_summary(summaries, step)
                return accuracy

            def overfit(dev_loss, accu):
                num = 6
                n = len(dev_loss)
                if n < num:
                    return False
                for i in xrange(n - num, n):
                    if dev_loss[i] < accu:
                        return False
                print(dev_loss)
                print(accu)
                return True

            def evaluate(x1_batch, x2_batch, x1_batch_m, x2_batch_m, y_batch, mention, entity):

                feed_dict = {
                    siameseModel.input_x1: x1_batch,
                    siameseModel.input_x2: x2_batch,
                    siameseModel.mask_x1: x1_batch_m,
                    siameseModel.mask_x2: x2_batch_m,
                    siameseModel.input_y: y_batch,
                }

                loss, accuracy, dist = sess.run([siameseModel.loss, siameseModel.accuracy, siameseModel.distance],
                                                feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("Test {}: loss {:g}, acc {:g}".format(time_str, loss, accuracy))
                print(dist)

                eval_file = open(out_dir + "/evaluation.txt", "w+")
                right_file = open(out_dir + "/right_cases.txt", "w+")
                wrong_file = open(out_dir + "/wrong_cases.txt", "w+")

                eval_file.write("Accu: " + str(accuracy) + "\n")
                eval_file.write("Dataset: " + data_file + "\n")
                eval_file.write("Early Stopped at: " + str(stop_p) + "\n")

                d = np.copy(dist)
                d[d >= 0.5] = 999.0
                d[d < 0.5] = 1
                d[d > 1.0] = 0

                predictions = d
                write_evaluation_file(eval_file, right_file, wrong_file, y_batch, predictions, mention, entity)
                return accuracy

            # Generate batches
            batches = inpH.batch_iter(list(zip(train_set[0], train_set[1], train_set[2], train_set[3], train_set[4])),
                                      FLAGS.batch_size, FLAGS.num_epochs)

            max_validation_acc = 0.0
            num_batches_per_epoch = int(len(train_set[0]) / FLAGS.batch_size)
            print num_batches_per_epoch
            max_accu = 0
            dev_accu = []

            for nn in xrange(sum_no_of_batches * FLAGS.num_epochs):
                batch = batches.next()
                if len(batch) < 1:
                    continue
                x1_batch, x2_batch, x1_batch_m, x2_match_m, y_batch = zip(*batch)
                if len(y_batch) < 1:
                    continue

                train_step(x1_batch, x2_batch, x1_batch_m, x2_match_m, y_batch)

                current_step = tf.train.global_step(sess, global_step)  # get the global step.
                sum_acc = 0.0
                tmp = []

                if current_step % num_batches_per_epoch == 0:
                    print("\nEvaluation:")
                    # dev_batches = inpH.batch_iter(list(zip(dev_set[0], dev_set[1], dev_set[2], dev_set[3], dev_set[4])),
                    #                               FLAGS.batch_size, 1)
                    # for db in dev_batches:
                    #     if len(db) < 1:
                    #         continue
                    #     x1_dev_b, x2_dev_b, x1_dev_m, x2_dev_m, y_dev_b = zip(*db)
                    #     if len(y_dev_b) < 1:
                    #         continue
                    #     acc = dev_step(x1_dev_b, x2_dev_b, x1_dev_m, x2_dev_m, y_dev_b)
                    #     sum_acc = sum_acc + acc
                    #     tmp.append(acc)
                    #
                    # acc_mean = np.mean(tmp)
                    acc_mean = dev_step(dev_set[0], dev_set[1], dev_set[2], dev_set[3], dev_set[4])
                    dev_accu.append(acc_mean)

                    if overfit(dev_accu, acc_mean):
                        print 'Overfit!!'
                        print("Optimum" + str(max_accu))
                        print(current_step)
                        stop_p = current_step / num_batches_per_epoch
                        print(stop_p)
                        break

                    if acc_mean >= max_accu:
                        max_accu = acc_mean
                        saver.save(sess, checkpoint_prefix, global_step=current_step)  # save checkpoints
                        tf.train.write_graph(sess.graph.as_graph_def(), checkpoint_prefix, "graph" + str(nn) + ".pb",
                                             as_text=False)  # save graph_def
                        print("Saved model {} with sum_accuracy={} checkpoint to {}\n".format(nn, max_validation_acc,
                                                                                              checkpoint_prefix))

            evaluate(test_set[0], test_set[1], test_set[2], test_set[3], test_set[4], test_x1, test_x2)


s = SentenceModel()
