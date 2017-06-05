#!/usr/bin/env python
# coding=utf-8
import numpy as np
import gensim, jieba, os
from sentence_model import getSentence_Embedding
from scipy import spatial
from DLDisambiguation.util.util import preprocess_arr
from DLDisambiguation.util.input_helpers import InputHelper


class Tensor(object):
    def __init__(self, m, e, batch_size, sequence_length):
        """
        initialize tensors
        :param m: mention list
        :param e: entity list
        :param batch_size: batch_size
        :param sequence_length: default sentence length
        """
        channel_num = 4
        dir = "/Users/luoyi/Documents/Python/DLDisambiguation/data/word2vec"
        character_model_file = os.path.join(dir, 'character.model')
        word_mode_file = os.path.join(dir, 'word.model')

        self.mentions = preprocess_arr(m)
        self.entities = preprocess_arr(e)
        self.sequence_length = sequence_length
        self.batch_size = batch_size

        self.character_embedding_model = gensim.models.Word2Vec.load(character_model_file)
        self.word_embedding_model = gensim.models.Word2Vec.load(word_mode_file)

        self.tensor = np.zeros(shape=(batch_size, channel_num, sequence_length, sequence_length))

        self.init_str_matrix()
        self.init_character_matrix()
        self.init_word_matrix()
        self.init_sentence_matrix()

    def init_str_matrix(self):
        """
        initialize the str-match matrix
        :return:
        """
        for sample_index in range(self.batch_size):
            len_mention = len(self.mentions[sample_index].decode("utf-8"))
            len_entity = len(self.entities[sample_index].decode("utf-8"))

            for i in range(len_mention):
                for j in range(len_entity):

                    if self.mentions[sample_index][i] == self.entities[sample_index][j]:
                        self.tensor[sample_index][0][i][j] = 1

    def init_character_matrix(self):
        """
        initialize the character-match matrix
        :return:
        """

        for sample_index in range(self.batch_size):
            len_mention = len(self.mentions[sample_index].decode("utf-8"))
            len_entity = len(self.entities[sample_index].decode("utf-8"))
            for i in range(len_mention):
                for j in range(len_entity):
                    character_embedding_mi = self.get_embedding(self.mentions[sample_index][i],
                                                                self.character_embedding_model)
                    character_embedding_ei = self.get_embedding(self.entities[sample_index][j],
                                                                self.character_embedding_model)
                    self.tensor[sample_index][1][i][j] = 1 - spatial.distance.cosine(character_embedding_ei,
                                                                                     character_embedding_mi)

    def init_word_matrix(self):
        """
        initialize the word-match matrix
        :return:
        """

        for sample_index in range(self.batch_size):
            words_m = list(jieba.cut(self.mentions[sample_index]))
            words_e = list(jieba.cut(self.entities[sample_index]))
            len_w_m = len(words_m)
            len_w_e = len(words_e)

            for i in range(len_w_m):
                for j in range(len_w_e):

                    words_sim = 1 - spatial.distance.cosine(
                        self.get_embedding(words_m[i], self.word_embedding_model),
                        self.get_embedding(words_e[j], self.word_embedding_model))

                    # assign the word_pair_sim to the character_pairs which construct the words
                    for character_i in words_m[i]:
                        for character_j in words_e[j]:
                            self.tensor[sample_index][2][
                                self.mentions[sample_index].index(character_i), self.entities[sample_index].index(
                                    character_j)] = words_sim

    def init_sentence_matrix(self):
        """
        initialize the sentence_level-match matrix
        :return:
        """
        sentence_embedding_m, sentence_embedding_e = getSentence_Embedding(self.mentions, self.entities)

        for sample_index in range(self.batch_size):
            for i in range(self.sequence_length):
                for j in range(self.sequence_length):
                    self.tensor[sample_index][3][i][j] = 1 - spatial.distance.cosine(
                        sentence_embedding_m[sample_index][i], sentence_embedding_e[sample_index][j])

    def get_tensor(self):
        return self.tensor

    def get_embedding(self, word, model):
        if word in model.wv.vocab.keys():
            index = model.wv.index2word.index(word)
            return model.wv.syn0[index]
            # return model.wv.vocab[word]
        else:
            vector_length = 100
            return np.ones([vector_length])

if __name__ == "__main__":
    # test Tensor construction

    file_path = "./data/validation.txt0"
    inpH = InputHelper()
    testx1, testx2, y = inpH.getTsvTestData(file_path)

    t = Tensor(testx1, testx2, len(testx1), 20)
    res = t.get_tensor()
    print(res)
