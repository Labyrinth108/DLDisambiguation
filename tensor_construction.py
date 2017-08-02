#!/usr/bin/env python
# encoding=utf-8

import datetime
import os
import time

import numpy as np
import tensorflow as tf

from DLDisambiguation.util.input_helpers import InputHelper
from tensor import Tensor

tf.flags.DEFINE_string("train_dir", "./", "Training dir root")
tf.flags.DEFINE_string("max_sequence_len", 10, "length")
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()


def generate_Tensor(mention, entity, mention2, entity2, mention3, entity3, max_len, task_n):
    # lstm_dir = "Description1501174251" if task_n == 1 else "Operation1501209225"
    lstm_dir = "Description1501554142" if task_n == 1 else "Operation1501588184"
    bilstm_dir = os.path.join("./Sentence_Modeling/runs", lstm_dir)

    men_arr = np.concatenate((mention, mention2, mention3))
    entity_arr = np.concatenate((entity, entity2, entity3))
    tensor = Tensor(men_arr, entity_arr, len(men_arr), max_len, task_n, bilstm_dir).get_tensor()
    tensor = tensor.transpose((0, 2, 3, 1))

    g1 = len(mention)
    g2 = len(np.concatenate((mention, mention2)))
    return tensor[:g1], tensor[g1:g2], tensor[g2:]


# def prepara_tensor_y(inputH, training_path, dev_path, test_path, max_len):
#     x1_train, x2_train, x3_train, x4_train, y_train, y2_train = inputH.getTsvTestData_Mul(training_path, "\t", max_len)
#     x1_dev, x2_dev, x3_dev, x4_dev, y_dev, y2_dev = inputH.getTsvTestData_Mul(dev_path, "\t", max_len)
#     x1_test, x2_test, x3_test, x4_test, y_test, y2_test = inputH.getTsvTestData_Mul(test_path, "\t", max_len)
#
#     print("Finished Loading")
#     x_train_tensor, x_dev_tensor, x_test_tensor = generate_Tensor(x1_train, x2_train, x1_dev, x2_dev, x1_test, x2_test,
#                                                                   max_len, 1)
#     print("Finished constructing tensors!")
#
#     np.save("train_des", x_train_tensor)
#     np.save("dev_des", x_dev_tensor)
#     np.save("test_des", x_test_tensor)
#     print("Save description tensors!")
#
#     x_train_tensor_o, x_dev_tensor_o, x_test_tensor_o = generate_Tensor(x3_train, x4_train, x3_dev, x4_dev, x3_test,
#                                                                         x4_test, max_len, 2)
#
#     np.save("train_opr", x_train_tensor_o)
#     np.save("dev_opr", x_dev_tensor_o)
#     np.save("test_opr", x_test_tensor_o)
#
#     return x_train_tensor, y_train, x_dev_tensor, y_dev, x_test_tensor, y_test, \
#            x_train_tensor_o, y2_train, x_dev_tensor_o, y2_dev, x_test_tensor_o, y2_test

def prepara_tensor_y_seperate(inputH, data_file, data_file_val, data_file_test, sep, max_len, name, task_num):
    x_train_mention, x_train_entity, y_train = inputH.getTsvTestData(data_file, sep, max_len, y_value=False)
    x_dev_mention, x_dev_entity, y_dev = inputH.getTsvTestData(data_file_val, sep, max_len, y_value=False)
    x_test_mention, x_test_entity, y_test = inputH.getTsvTestData(data_file_test, sep, max_len, y_value=False)

    print("Finished Loading")
    x_train_tensor, x_dev_tensor, x_test_tensor = generate_Tensor(x_train_mention, x_train_entity, x_dev_mention,
                                                                  x_dev_entity, x_test_mention, x_test_entity,
                                                                  max_len, task_num)
    print("Finished constructing tensors!")

    print("Length")
    print(len(x_train_mention))
    mydir = "Length10/"
    np.save(mydir + "train_" + name, x_train_tensor)
    np.save(mydir + "dev_" + name, x_dev_tensor)
    np.save(mydir + "test_" + name, x_test_tensor)
    print("Save tensors!")


def main():
    # Load data
    print("Loading data...")
    inputH = InputHelper()

    # train_f = os.path.join(FLAGS.train_dir, 'data/exp/training_data_0724.txt')
    # dev_f = os.path.join(FLAGS.train_dir, 'data/exp/validation_data_0724.txt')
    # test_f = os.path.join(FLAGS.train_dir, 'data/exp/test_data_0724.txt')

    # data_file = os.path.join(FLAGS.train_dir, "data/test01.txt")
    # data_file_val = os.path.join(FLAGS.train_dir, "data/test01.txt")
    # data_file_test = os.path.join(FLAGS.train_dir, "data/test01.txt")

    task_num = 2
    name = "des" if task_num == 1 else "opr"

    data_file = os.path.join(FLAGS.train_dir, "data/training_data_0724_" + name + ".txt")
    data_file_test = os.path.join(FLAGS.train_dir, "data/test_data_0724_" + name + ".txt")
    data_file_val = os.path.join(FLAGS.train_dir, "data/validation_data_0724_" + name + ".txt")

    # prepara_tensor_y(inputH, train_f, dev_f, test_f, FLAGS.max_sequence_len)
    prepara_tensor_y_seperate(inputH, data_file, data_file_val, data_file_test, "\t", FLAGS.max_sequence_len, name,
                              task_num)


if __name__ == '__main__':
    main()

    # arr = np.load('train_opr.npy')
    # print(arr)
