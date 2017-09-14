#!/usr/bin/env python
# encoding=utf-8

import os
import numpy as np
import tensorflow as tf

from DLDisambiguation.util.input_helpers import InputHelper
from tensor import Tensor

tf.flags.DEFINE_string("train_dir", "./", "Training dir root")
tf.flags.DEFINE_string("max_sequence_len", 10, "length")
tf.flags.DEFINE_string("max_sequence_len2", 20, "length")
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()


def generate_Tensor(mention, entity, mention2, entity2, mention3, entity3, max_len, task_n):
    lstm_dir = "Description1502868912" if task_n == 1 else "Operation1502954903"  # 0810数据集 10 + 20
    # lstm_dir = "Description1501554142" if task_n == 1 else "Operation1501588184"  # 旧数据集 8万 LEN = 10
    bilstm_dir = os.path.join("./Sentence_Modeling/Exp/runs", lstm_dir)

    men_arr = np.concatenate((mention, mention2, mention3))
    entity_arr = np.concatenate((entity, entity2, entity3))
    tensor = Tensor(men_arr, entity_arr, len(men_arr), max_len, task_n, bilstm_dir).get_tensor()
    tensor = tensor.transpose((0, 2, 3, 1))

    g1 = len(mention)
    g2 = len(np.concatenate((mention, mention2)))
    return tensor[:g1], tensor[g1:g2], tensor[g2:]


def generate_Tensor_no_dev(mention, entity, mention3, entity3, max_len, task_n):
    lstm_dir = "Description1503482587" if task_n == 1 else "Operation1503500979"  # 0823
    # lstm_dir = "Description1503276064" if task_n == 1 else "Operation1503277387"  # 0816
    # lstm_dir = "Description1503227493" if task_n == 1 else "Operation1502964494"  #
    # lstm_dir = "Description1502964352" if task_n == 1 else "Operation1502964494"  # 0810数据集 10 + 20
    print lstm_dir
    bilstm_dir = os.path.join("./Sentence_Modeling/Exp0823/runs", lstm_dir)

    men_arr = np.concatenate((mention, mention3))
    entity_arr = np.concatenate((entity, entity3))
    tensor = Tensor(men_arr, entity_arr, len(men_arr), max_len, task_n, bilstm_dir).get_tensor()
    tensor = tensor.transpose((0, 2, 3, 1))

    g1 = len(mention)
    return tensor[:g1], tensor[g1:]


def prepara_dynamic_tensor(inputH, training_path, dev_path, test_path, max_len, max_len2):
    indi1, x1_train, x2_train, x3_train, x4_train, y_train, y2_train = inputH.getTsvTestData_Mul_Labels_Dyna(
        training_path, "\t", max_len)
    indi2, x1_dev, x2_dev, x3_dev, x4_dev, y_dev, y2_dev = inputH.getTsvTestData_Mul_Labels_Dyna(dev_path, "\t",
                                                                                                 max_len)
    indi3, x1_test, x2_test, x3_test, x4_test, y_test, y2_test = inputH.getTsvTestData_Mul_Labels_Dyna(test_path, "\t",
                                                                                                       max_len)

    print("Finished Loading")
    x_train_tensor, x_dev_tensor, x_test_tensor = generate_Tensor(x1_train, x2_train, x1_dev, x2_dev, x1_test, x2_test,
                                                                  max_len, 1)
    print("Finished constructing tensors!")

    dir_t = "./0816/"
    np.save(dir_t + "train_des", x_train_tensor)
    np.save(dir_t + "dev_des", x_dev_tensor)
    np.save(dir_t + "test_des", x_test_tensor)
    print("Save description tensors!")

    x_train_tensor_o, x_dev_tensor_o, x_test_tensor_o = generate_Tensor(x3_train, x4_train, x3_dev, x4_dev, x3_test,
                                                                        x4_test, max_len2, 2)

    np.save(dir_t + "train_opr", x_train_tensor_o)
    np.save(dir_t + "dev_opr", x_dev_tensor_o)
    np.save(dir_t + "test_opr", x_test_tensor_o)

    np.save(dir_t + "train_indi_opr", indi1)
    np.save(dir_t + "dev_indi_opr", indi2)
    np.save(dir_t + "test_indi_opr", indi3)

def prepara_tensor_y_seperate(inputH, data_file, data_file_val, data_file_test, sep, max_len, name, task_num):
    x_train_mention, x_train_entity, y_train = inputH.getTsvTestData(data_file, sep, max_len, y_value=False)
    # x_dev_mention, x_dev_entity, y_dev = inputH.getTsvTestData(data_file_val, sep, max_len, y_value=False)
    x_test_mention, x_test_entity, y_test = inputH.getTsvTestData(data_file_test, sep, max_len, y_value=False)

    print("Finished Loading")
    # x_train_tensor, x_dev_tensor, x_test_tensor = generate_Tensor(x_train_mention, x_train_entity, x_dev_mention,
    #                                                               x_dev_entity, x_test_mention, x_test_entity,
    #                                                               max_len,  task_num)
    x_train_tensor, x_test_tensor = generate_Tensor_no_dev(x_train_mention, x_train_entity,
                                                                         x_test_mention, x_test_entity,
                                                                         max_len, task_num)
    print("Finished constructing tensors!")

    print("Length")
    print(len(x_train_mention))
    mydir = "Tensor_files/0823/No_IDF/Length" + str(max_len) + "/"
    np.save(mydir + "train_" + name, x_train_tensor)
    # np.save(mydir + "dev_" + name, x_dev_tensor)
    np.save(mydir + "test_" + name, x_test_tensor)
    print("Save tensors!")


def main():
    # Load data
    print("Loading data...")
    inputH = InputHelper()

    task_num = 1
    name = "des" if task_num == 1 else "opr"

    # train_f = "./data/exp0803/training_dynamic_data.txt"
    # dev_f = "./data/exp0803/validation_dynamic_data.txt"
    # test_f = "./data/exp0803/test_dynamic_data.txt"
    # prepara_dynamic_tensor(inputH, train_f, dev_f, test_f, FLAGS.max_sequence_len, FLAGS.max_sequence_len2)

    time_gen = "0823"
    data_file = os.path.join(FLAGS.train_dir, "data/train_data_" + time_gen + "_" + name + ".txt")
    data_file_test = os.path.join(FLAGS.train_dir, "data/test_data_" + time_gen + "_" + name + ".txt")
    data_file_val = data_file_test
    # data_file_val = os.path.join(FLAGS.train_dir, "data/validation_data_" + time_gen + "_" + name + ".txt")
    prepara_tensor_y_seperate(inputH, data_file, data_file_val, data_file_test, "\t", FLAGS.max_sequence_len, name,
                              task_num)


if __name__ == '__main__':
    main()
