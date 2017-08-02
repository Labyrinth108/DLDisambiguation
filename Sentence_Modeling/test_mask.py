from DLDisambiguation.util.input_helpers import InputHelper
from DLDisambiguation.util.preprocess import MyVocabularyProcessor
import tensorflow as tf
import os
import numpy as np


def writeToFile(m, file):
    l = len(m)
    for i in range(l):
        l_col = len(m[i])
        file.write("[")
        for j in range(l_col):
            file.write(str(m[i][j]) + " ")
        file.write("]\n")
    file.write("\n")

inpH = InputHelper()
max_document_length = 10
y_is_value = True


def get_data(vocab_processor, inpH, train_x1, train_x2, train_y, max_document_length):
    train_x1_i = np.asarray(list(vocab_processor.transform(train_x1)))
    train_x2_i = np.asarray(list(vocab_processor.transform(train_x2)))

    mask_train_x1 = np.zeros([len(train_x1_i), max_document_length])
    mask_train_x2 = np.zeros([len(train_x2_i), max_document_length])

    new_mask_x1, new_mask_x2 = inpH.padding_and_generate_mask(train_x1, train_x2, mask_train_x1, mask_train_x2)
    return (train_x1_i, train_x2_i, new_mask_x1, new_mask_x2, train_y)


model_dir = "./runs/Operation1501588184"

checkpoint_dir = os.path.join(model_dir, "checkpoints")
print(checkpoint_dir)
ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
checkpoint_file = ckpt.model_checkpoint_path

vocab_file = os.path.join(checkpoint_dir, "vocab")
# load vocabulary model
vocab_processor = MyVocabularyProcessor(max_document_length, min_frequency=0)
vocab_processor = vocab_processor.restore(vocab_file)

task_num = 2
name = "des" if task_num == 1 else "opr"
test_file = "../data/test_data_0724_" + name + ".txt"
test_x1, test_x2, test_y = inpH.getTsvTestData(test_file, "\t", max_document_length, y_is_value)

test_set = get_data(vocab_processor, inpH, test_x1, test_x2, test_y, max_document_length)

filename = model_dir + "/test_look.txt"
file = open(filename, "w+")

param_f = open(model_dir + "/params.txt", "w+")
graph = tf.Graph()

with graph.as_default():
    sess = tf.Session()
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        sess.run(tf.initialize_all_variables())
        saver.restore(sess, checkpoint_file)

        # for var in tf.trainable_variables():
        #     print(var.name)
        #     print(var)
        #     print(sess.run(var))

        # Get the placeholders from the graph by name
        # the output is a list with only one element
        input_x1 = graph.get_operation_by_name("input_x1").outputs[0]
        mask_x1 = graph.get_operation_by_name("mask_x1").outputs[0]
        input_x2 = graph.get_operation_by_name("input_x2").outputs[0]
        mask_x2 = graph.get_operation_by_name("mask_x2").outputs[0]
        input_y = graph.get_operation_by_name("input_y").outputs[0]

        accuracy = graph.get_operation_by_name("accuracy/accuracy").outputs[0]
        r1 = graph.get_operation_by_name("sentence_embedding/Representation1").outputs[0]
        r2 = graph.get_operation_by_name("sentence_embedding/Representation2").outputs[0]

        accu, vr1, vr2 = sess.run([accuracy, r1, r2], {input_x1: test_set[0], mask_x1: test_set[2],
                                                       input_x2: test_set[1], mask_x2: test_set[3],
                                                       input_y: test_set[4]})

    # r(input_size, None, hidden_n * 2) => (None, input_size, hidden_n * 2)
    representation1 = np.transpose(vr1, (1, 0, 2))
    representation2 = np.transpose(vr2, (1, 0, 2))

    test_x1 = list(test_x1)
    test_x2 = list(test_x2)
    test_y = list(test_y)
    sample_n = len(representation1)
    for i in range(sample_n):
        matrix1 = representation1[i]
        matrix2 = representation2[i]
        file.write("sample " + str(i) + "\n")
        file.write(str(test_x1[i]) + " " + str(test_x2[i]) + " " + str(test_y[i]) + "\n")
        writeToFile(matrix1, file)
        writeToFile(matrix2, file)

        file.write("\n")

    print(accu)
