#!/usr/bin/env python
# coding=utf-8

import codecs
import gc
import sys

import numpy as np

from preprocess import MyVocabularyProcessor
from util import preprocess_unit, preprocess_arr

reload(sys)
sys.setdefaultencoding("utf-8")


class InputHelper(object):
    def get_data(self, vocab_processor, train_x1, train_x2, train_y, max_document_length):
        """
        Use vocab_processor to index mention and entity pairs and then pad them and return mask arrs
        :param vocab_processor:
        :param train_x1:
        :param train_x2:
        :param train_y:
        :param max_document_length:
        :return:
        """
        train_x1_i = np.asarray(list(vocab_processor.transform(train_x1)))
        train_x2_i = np.asarray(list(vocab_processor.transform(train_x2)))

        mask_train_x1 = np.zeros([len(train_x1_i), max_document_length])
        mask_train_x2 = np.zeros([len(train_x2_i), max_document_length])

        new_mask_x1, new_mask_x2 = self.padding_and_generate_mask(train_x1, train_x2, mask_train_x1, mask_train_x2)
        return (train_x1_i, train_x2_i, new_mask_x1, new_mask_x2, train_y)

    def padding_and_generate_mask(self, x1, x2, new_mask_x1, new_mask_x2):
        """
        Pad the sentence and return mask array for mention and entity pair
        :param x1:
        :param x2:
        :param new_mask_x1:
        :param new_mask_x2:
        :return:
        """

        for i, (x1, x2) in enumerate(zip(x1, x2)):
            # whether to remove sentences with length larger than maxlen
            if len(x1) == 0 or len(x2) == 0:
                print("")
            new_mask_x1[i, 0:len(x1)] = 1.0
            new_mask_x2[i, 0:len(x2)] = 1.0
        return new_mask_x1, new_mask_x2

    def add_y_helper(self, y_value, y_arr, is_positive_label):
        """
        add 1/0 or [0,1]/[1, 0] in y_arr which depends on y_value flag
        :param y_value:
        :param y_arr:
        :param is_positive_label:
        :return:
        """
        if y_value == True:
            if is_positive_label:
                y_arr.append(1)
            else:
                y_arr.append(0)
        else:
            if is_positive_label:
                y_arr.append(np.array([0, 1]))
            else:
                y_arr.append(np.array([1, 0]))
        return y_arr

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

        if shuffle:
            print "Shuffle!!!!"
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

    def getTestIndexedDataSet(self, data_path, sep, vocab_processor, max_document_length, y_value):
        """
        Read in labeled test data and use previous vocabulary processor to index them
        :param data_path:
        :param sep:
        :param vocab_processor:
        :param max_document_length:
        :param y_value:
        :return:
        """
        x1_temp, x2_temp, y = self.getTsvTestData(data_path, sep, max_document_length, y_value)

        x1 = np.asarray(list(vocab_processor.transform(x1_temp)))
        x2 = np.asarray(list(vocab_processor.transform(x2_temp)))
        return x1, x2, y

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
        print(len(vocab_processor.vocabulary_))

        datax1 = preprocess_arr(datax1)
        datax2 = preprocess_arr(datax2)
        x1 = np.asarray(list(vocab_processor.transform(datax1)))
        x2 = np.asarray(list(vocab_processor.transform(datax2)))
        # Randomly shuffle data
        del vocab_processor
        gc.collect()
        return x1, x2

    def getTsvTestData(self, filepath, sep, sequence_length, y_value=False):
        """
        load the data(label, mention, entity) from labeled files
        :param filepath:
        :return:  three lists(label_list, mention_list, entity_list)
        """
        print("Loading testing/labelled data from " + filepath)
        x1, x2 = [], []
        y = []
        line_num = 0
        for line in codecs.open(filepath, "r", "utf-8"):
            line_num += 1
            l = line.strip().split(sep)
            if len(l) < 3:
                continue

            l[1] = preprocess_unit(l[1])
            l[2] = preprocess_unit(l[2])

            if len(l[1]) == 0 or len(l[2]) == 0:
                continue

            # truncate when length is bigger than the max_length
            if len(l[1]) > sequence_length or len(l[2]) > sequence_length:
                l[1] = l[1][:sequence_length]
                l[2] = l[2][:sequence_length]

            x1.append(l[1])
            x2.append(l[2])
            y = self.add_y_helper(y_value, y, int(l[0]) == 1)
            if line_num != len(y):
                print("ei")
        print(line_num)
        return np.asarray(x1), np.asarray(x2), np.asarray(y)

    def getTsvTestData_Mul(self, filepath, sep, sequence_length, y_value=False):
        """
        load the data(label, mention, entity) from labeled mutlti-task files
        :param filepath:
        :return:  three lists(label_list, mention_list, entity_list)
        """
        print("Loading testing/labelled data from " + filepath)
        x1, x2, x3, x4 = [], [], [], []
        y = []
        y2 = []
        for line in codecs.open(filepath, "r", "utf-8"):
            l = line.strip().split(sep)
            if len(l) < 5:
                continue

            l[1] = preprocess_unit(l[1])
            l[2] = preprocess_unit(l[2])
            l[3] = preprocess_unit(l[3])
            l[4] = preprocess_unit(l[4])

            # truncate when length is bigger than the max_length
            if len(l[1]) > sequence_length or len(l[2]) > sequence_length or len(l[3]) > sequence_length or len(
                    l[4]) > sequence_length:
                l[1] = l[1][:sequence_length]
                l[2] = l[2][:sequence_length]
                l[3] = l[3][:sequence_length]
                l[4] = l[4][:sequence_length]

            x1.append(l[1])
            x2.append(l[2])
            x3.append(l[3])
            x4.append(l[4])
            y = self.add_y_helper(y_value, y, int(l[0]) == 1)
            y2 = self.add_y_helper(y_value, y2, int(l[0]) == 1)
        return np.asarray(x1), np.asarray(x2), np.asarray(x3), np.asarray(x4), np.asarray(y), np.asarray(y2)

    def getTsvTestData_Mul(self, filepath, sep, sequence_length, y_value=False):
        """
        load the data(label, mention, entity) from labeled mutlti-task files
        :param filepath:
        :return:  three lists(label_list, mention_list, entity_list)
        """
        print("Loading testing/labelled data from " + filepath)
        x1, x2, x3, x4 = [], [], [], []
        y = []
        y2 = []
        for line in codecs.open(filepath, "r", "utf-8"):
            l = line.strip().split(sep)
            if len(l) < 5:
                continue

            l[1] = preprocess_unit(l[1])
            l[2] = preprocess_unit(l[2])
            l[3] = preprocess_unit(l[3])
            l[4] = preprocess_unit(l[4])

            # truncate when length is bigger than the max_length
            if len(l[1]) > sequence_length or len(l[2]) > sequence_length or len(l[3]) > sequence_length or len(
                    l[4]) > sequence_length:
                l[1] = l[1][:sequence_length]
                l[2] = l[2][:sequence_length]
                l[3] = l[3][:sequence_length]
                l[4] = l[4][:sequence_length]

            x1.append(l[1])
            x2.append(l[2])
            x3.append(l[3])
            x4.append(l[4])
            y = self.add_y_helper(y_value, y, int(l[0]) == 1)
            y2 = self.add_y_helper(y_value, y2, int(l[0]) == 1)
        return np.asarray(x1), np.asarray(x2), np.asarray(x3), np.asarray(x4), np.asarray(y), np.asarray(y2)

    def getTsvTestData_Mul_Labels(self, filepath, sep, sequence_length, y_value=False):
        """
        load the data(label, mention, entity) from labeled mutlti-task files
        :param filepath:
        :return:  three lists(label_list, mention_list, entity_list)
        """
        print("Loading testing/labelled data from " + filepath)
        x1, x2, x3, x4 = [], [], [], []
        y = []
        y2 = []
        for line in codecs.open(filepath, "r", "utf-8"):
            l = line.strip().split(sep)
            if len(l) < 6:
                continue

            l[1] = preprocess_unit(l[1])
            l[2] = preprocess_unit(l[2])
            l[4] = preprocess_unit(l[4])
            l[5] = preprocess_unit(l[5])

            # truncate when length is bigger than the max_length
            if len(l[1]) > sequence_length or len(l[2]) > sequence_length or len(l[4]) > sequence_length or len(
                    l[5]) > sequence_length:
                l[1] = l[1][:sequence_length]
                l[2] = l[2][:sequence_length]
                l[5] = l[5][:sequence_length]
                l[4] = l[4][:sequence_length]

            x1.append(l[1])
            x2.append(l[2])
            x3.append(l[4])
            x4.append(l[5])
            y = self.add_y_helper(y_value, y, int(l[0]) == 1)
            y2 = self.add_y_helper(y_value, y2, int(l[3]) == 1)

        return np.asarray(x1), np.asarray(x2), np.asarray(x3), np.asarray(x4), np.asarray(y), np.asarray(y2)

    def getTsvTestData_Mul_Labels_Dyna(self, filepath, sep, sequence_length, y_value=False):
        """
        load the data(label, mention, entity) from labeled mutlti-task files
        :param filepath:
        :return:  three lists(label_list, mention_list, entity_list)
        """
        print("Loading testing/labelled data from " + filepath)
        x1, x2, x3, x4 = [], [], [], []
        y = []
        y2 = []
        indicate = []
        for line in codecs.open(filepath, "r", "utf-8"):
            l = line.strip().split(sep)
            l[1] = preprocess_unit(l[1])
            l[2] = preprocess_unit(l[2])
            if len(l[1]) > sequence_length or len(l[2]) > sequence_length:
                l[1] = l[1][:sequence_length]
                l[2] = l[2][:sequence_length]
            x1.append(l[1])
            x2.append(l[2])
            y = self.add_y_helper(y_value, y, int(l[0]) == 1)

            if len(l) == 3:  # dynamic single task1
                x3.append("")
                x4.append("")
                y2 = self.add_y_helper(y_value, y2, False)
                indicate.append(1)
            else:
                l[4] = preprocess_unit(l[4])
                l[5] = preprocess_unit(l[5])
                # truncate when length is bigger than the max_length
                if len(l[4]) > sequence_length or len(l[5]) > sequence_length:
                    l[5] = l[5][:sequence_length]
                    l[4] = l[4][:sequence_length]
                x3.append(l[4])
                x4.append(l[5])
                indicate.append(0)
                y2 = self.add_y_helper(y_value, y2, int(l[3]) == 1)

        return indicate, np.asarray(x1), np.asarray(x2), np.asarray(x3), np.asarray(x4), np.asarray(y), np.asarray(y2)
