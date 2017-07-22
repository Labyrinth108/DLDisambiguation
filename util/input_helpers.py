#!/usr/bin/env python
# coding=utf-8

import codecs
import gc
import sys
from random import random

import numpy as np

from DLDisambiguation.util.preprocess import MyVocabularyProcessor
from util import preprocess_unit, preprocess_arr

reload(sys)
sys.setdefaultencoding("utf-8")


class InputHelper(object):
    def getTsvData(self, filepath, sequence_length):
        """
        get positive data from file and do negative sampling
        :param filepath: the file with positive mapping
        :return:
        """
        print("Loading training data from " + filepath)
        x1 = []
        x2 = []
        y = []

        # positive samples from file
        for line in codecs.open(filepath, "r", "utf-8"):
            l = line.strip().split("\t")
            if len(l) < 2:
                continue

            if len(l[0]) > sequence_length or len(l[1]) > sequence_length:
                continue

            l[0] = preprocess_unit(l[0])
            l[1] = preprocess_unit(l[1])

            x1.append(l[0])
            x2.append(l[1])
            y.append(np.array([0, 1]))

        x2_s = np.asarray(x2)
        nega_len = len(x2_s)
        shuffle_indices = np.random.permutation(np.arange(nega_len))
        x2_s = x2_s[shuffle_indices]

        for i in xrange(nega_len):
            x1.append(x1[i])
            x2.append(x2_s[i])
            y.append(np.array([1, 0]))

        return np.asarray(x1), np.asarray(x2), np.asarray(y)

    def getTsv_MultiData(self, filepath, sequence_length):
        """
        get positive data from file and do negative sampling
        :param filepath: the file with positive mapping
        :return:
        """
        print("Loading training data from " + filepath)
        x1, x2, x3, x4 = [], [], [], []
        y, y2 = [], []

        # positive samples from file
        for line in codecs.open(filepath, "r", "utf-8"):
            l = line.strip().split("\t")
            if len(l) < 4:
                continue

            if len(l[0]) > sequence_length or len(l[1]) > sequence_length or len(l[2]) > sequence_length or len(
                    l[3]) > sequence_length:
                continue

            l[0] = preprocess_unit(l[0])
            l[1] = preprocess_unit(l[1])
            l[2] = preprocess_unit(l[2])
            l[3] = preprocess_unit(l[3])

            x1.append(l[0])
            x2.append(l[1])
            x3.append(l[2])
            x4.append(l[3])
            y.append(np.array([0, 1]))
            y2.append(np.array([0, 1]))

        # generate random negative samples
        shuffle_indices = np.random.permutation(np.arange(len(x1)))
        x2_s = np.asarray(x2)
        x4_s = np.asarray(x4)
        x2_s = x2_s[shuffle_indices]
        x4_s = x4_s[shuffle_indices]
        nega_len = len(shuffle_indices)

        for i in xrange(nega_len):
            x1.append(x1[i])
            x3.append(x3[i])
            x2.append(x2_s[i])
            x4.append(x4_s[i])
            y.append(np.array([1, 0]))
            y2.append(np.array([1, 0]))

        return np.asarray(x1), np.asarray(x2), np.asarray(x3), np.asarray(x4), np.asarray(y), np.asarray(y2)

    def batch_iter(self, data, batch_size, num_epochs, shuffle=True):
        """
        Generates a batch iterator for a data set.
        :param data:
        :param batch_size:
        :param num_epochs:
        :param shuffle:
        :return:
        """
        data = np.asarray(data)
        print(data)
        print(data.shape)
        data_size = len(data)
        num_batches_per_epoch = int(len(data) / batch_size)

        for epoch in range(num_epochs):
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
            else:
                shuffled_data = data
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield shuffled_data[start_index:end_index]

    def dumpValidation(self, x1_text, x2_text, y, shuffled_index, dev_idx, i):
        print("dumping validation " + str(i))
        x1_shuffled = x1_text[shuffled_index]
        x2_shuffled = x2_text[shuffled_index]
        y_shuffled = y[shuffled_index]
        x1_dev = x1_shuffled[dev_idx:]
        x2_dev = x2_shuffled[dev_idx:]
        y_dev = y_shuffled[dev_idx:]
        del x1_shuffled
        del y_shuffled

        with open('validation.txt' + str(i), 'w') as f:
            for text1, text2, label in zip(x1_dev, x2_dev, y_dev):
                f.write(str(label) + "\t\t" + text1 + "\t\t" + text2 + "\n")
            f.close()
        del x1_dev
        del y_dev

    def getDataSets(self, training_paths, max_document_length, percent_dev, batch_size):
        x1_text, x2_text, y = self.getTsvData(training_paths, max_document_length)

        # Build vocabulary
        print("Building vocabulary")
        vocab_processor = MyVocabularyProcessor(max_document_length, min_frequency=0)
        vocab_processor.fit_transform(np.concatenate((x2_text, x1_text), axis=0))
        print("Length of loaded vocabulary ={}".format(len(vocab_processor.vocabulary_)))

        x1 = np.asarray(list(vocab_processor.transform(x1_text)))
        x2 = np.asarray(list(vocab_processor.transform(x2_text)))

        # Randomly shuffle data
        np.random.seed(131)
        shuffle_indices = np.random.permutation(np.arange(len(y)))
        x1_shuffled = x1[shuffle_indices]
        x2_shuffled = x2[shuffle_indices]
        y_shuffled = y[shuffle_indices]
        dev_idx = -1 * len(y_shuffled) * percent_dev // 100
        del x1, x2

        # Split train/test set
        # self.dumpValidation(x1_text, x2_text, y, shuffle_indices, dev_idx, 0)

        # TODO: This is very crude, should use cross-validation
        x1_train, x1_dev = x1_shuffled[:dev_idx], x1_shuffled[dev_idx:]
        x2_train, x2_dev = x2_shuffled[:dev_idx], x2_shuffled[dev_idx:]
        y_train, y_dev = y_shuffled[:dev_idx], y_shuffled[dev_idx:]
        print("Train/Dev split for {}: {:d}/{:d}".format(training_paths, len(y_train), len(y_dev)))

        sum_no_of_batches = len(y_train) // batch_size
        train_set = (x1_train, x2_train, y_train)
        dev_set = (x1_dev, x2_dev, y_dev)
        gc.collect()
        return train_set, dev_set, vocab_processor, sum_no_of_batches

    def splitDataSets(self, training_paths, percent_dev, max_len):
        x1_text, x2_text, y = self.getTsvData(training_paths, max_len)

        # Randomly shuffle data
        np.random.seed(131)
        shuffle_indices = np.random.permutation(np.arange(len(y)))
        x1_shuffled = x1_text[shuffle_indices]
        x2_shuffled = x2_text[shuffle_indices]
        y_shuffled = y[shuffle_indices]
        dev_idx = -1 * len(y_shuffled) * percent_dev // 100

        # Split train/test set
        # self.dumpValidation(x1_text, x2_text, y, shuffle_indices, dev_idx, 0)

        # TODO: This is very crude, should use cross-validation
        x1_train, x1_dev = x1_shuffled[:dev_idx], x1_shuffled[dev_idx:]
        x2_train, x2_dev = x2_shuffled[:dev_idx], x2_shuffled[dev_idx:]
        y_train, y_dev = y_shuffled[:dev_idx], y_shuffled[dev_idx:]
        print("Train/Dev split for {}: {:d}/{:d}".format(training_paths, len(y_train), len(y_dev)))

        gc.collect()
        return x1_train, x2_train, y_train, x1_dev, x2_dev, y_dev

    def splitMulti_DataSets(self, training_paths, percent_dev, max_len):
        x1_text, x2_text, x3_text, x4_text, y, y2 = self.getTsv_MultiData(training_paths, max_len)

        # Randomly shuffle data
        np.random.seed(131)
        shuffle_indices = np.random.permutation(np.arange(len(y)))
        x1_shuffled = x1_text[shuffle_indices]
        x2_shuffled = x2_text[shuffle_indices]
        x3_shuffled = x3_text[shuffle_indices]
        x4_shuffled = x4_text[shuffle_indices]
        y_shuffled = y[shuffle_indices]
        y2_shuffled = y2[shuffle_indices]
        dev_idx = -1 * len(y_shuffled) * percent_dev // 100

        # Split train/test set
        # self.dumpValidation(x1_text, x2_text, y, shuffle_indices, dev_idx, 0)

        # TODO: This is very crude, should use cross-validation
        x1_train, x1_dev = x1_shuffled[:dev_idx], x1_shuffled[dev_idx:]
        x2_train, x2_dev = x2_shuffled[:dev_idx], x2_shuffled[dev_idx:]
        x3_train, x3_dev = x3_shuffled[:dev_idx], x3_shuffled[dev_idx:]
        x4_train, x4_dev = x4_shuffled[:dev_idx], x4_shuffled[dev_idx:]
        y_train, y_dev = y_shuffled[:dev_idx], y_shuffled[dev_idx:]
        y2_train, y2_dev = y2_shuffled[:dev_idx], y2_shuffled[dev_idx:]
        print("Train/Dev split for {}: {:d}/{:d}".format(training_paths, len(y_train), len(y_dev)))

        gc.collect()
        return x1_train, x2_train, x3_train, x4_train, y_train, y2_train, x1_dev, x2_dev, x3_dev, x4_dev, y_dev, y2_dev

    def getTestDataSet(self, data_path, vocab_path, max_document_length):
        x1_temp, x2_temp, y = self.getTsvTestData(data_path)

        # Build vocabulary
        vocab_processor = MyVocabularyProcessor(max_document_length, min_frequency=0)
        vocab_processor = vocab_processor.restore(vocab_path)
        print len(vocab_processor.vocabulary_)

        x1 = np.asarray(list(vocab_processor.transform(x1_temp)))
        x2 = np.asarray(list(vocab_processor.transform(x2_temp)))
        # Randomly shuffle data
        del vocab_processor
        gc.collect()
        return x1_temp, x2_temp, x1, x2, y

    def toVocabularyIndexVector(self, datax1, datax2, vocab_path, max_document_length):
        """
        Transform the word list to vocabulary_index vectors
        :param datax1:
        :param datax2:
        :param vocab_path:
        :param max_document_length:
        :return:
        """
        # Build vocabulary
        vocab_processor = MyVocabularyProcessor(max_document_length, min_frequency=0)
        vocab_processor = vocab_processor.restore(vocab_path)
        print len(vocab_processor.vocabulary_)

        datax1 = preprocess_arr(datax1)
        datax2 = preprocess_arr(datax2)
        x1 = np.asarray(list(vocab_processor.transform(datax1)))
        x2 = np.asarray(list(vocab_processor.transform(datax2)))
        # Randomly shuffle data
        del vocab_processor
        gc.collect()
        return x1, x2

    def getTsvTestData(self, filepath):
        """
        load the data(label, mention, entity) from test file and
        :param filepath:
        :return:  three lists(mention_list, entity_list, label_list)
        """
        print("Loading testing/labelled data from " + filepath)
        x1 = []
        x2 = []
        y = []
        # positive samples from file
        for line in codecs.open(filepath, "r", "utf-8"):
            l = line.strip().split("\t\t")
            if len(l) < 3:
                continue
            l[1] = preprocess_unit(l[1])
            l[2] = preprocess_unit(l[2])
            x1.append(l[1])
            x2.append(l[2])
            y.append(int(l[0]))  # np.array([0,1]))
        return np.asarray(x1), np.asarray(x2), np.asarray(y)