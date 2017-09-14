#!/usr/bin/env python
# coding=utf-8
import gensim, jieba, os
from scipy import spatial
from util.util import preprocess_arr
import numpy as np
import tensorflow as tf

from util.input_helpers import InputHelper
from util.preprocess import MyVocabularyProcessor
from util.util import loadIDFModel


class Tensor(object):
    def __init__(self, m, e, batch_size, sequence_length, task_num, lstm_dir):
        """
        initialize tensors
        :param m: mention list
        :param e: entity list
        :param batch_size: batch_size
        :param sequence_length: default sentence length
        :param task_num: the type of task(1--description, 2--operation)
        """
        channel_num = 4
        if task_num == 1:  # task1-description disambiguation
            word2vec_dir = "./data/word2vec"
            self.idfModel_file = "./data/description_idf.txt"
            # self.idfModel_file = "./data/idfModel.txt"

        else:  # task2-operation disambiguation
            word2vec_dir = "./data/operation"
            self.idfModel_file = "./data/operation_idf.txt"
            # self.idfModel_file = "./data/idfModel_operation.txt"

        character_model_file = os.path.join(word2vec_dir, 'character.model')
        word_mode_file = os.path.join(word2vec_dir, 'word.model')
        self.bilstm_dir = lstm_dir

        self.mentions = preprocess_arr(m)
        self.entities = preprocess_arr(e)
        self.sequence_length = sequence_length
        self.batch_size = batch_size

        self.character_embedding_model = gensim.models.Word2Vec.load(character_model_file)
        self.word_embedding_model = gensim.models.Word2Vec.load(word_mode_file)

        self.tensor = np.zeros(shape=(batch_size, channel_num, sequence_length, sequence_length))
        self.init_matrices()

    def init_matrices(self):
        """
        initialize four matrices in the tensor
        :return:
        """
        sentence_embedding_m, sentence_embedding_e = self.getSentence_Embedding(self.mentions, self.entities,
                                                                                self.sequence_length)
        np.save("no_att_sentece_m_opr.npy", sentence_embedding_m)
        np.save("no_att_sentece_e_opr.npy", sentence_embedding_e)
        # sentence_embedding_m = np.load("0823_sentence_m.npy")
        # sentence_embedding_e = np.load("0823_sentence_e.npy")

        print("Sentence Embedding Finished!")

        for sample_index in range(self.batch_size):
            len_mention = len(self.mentions[sample_index].decode("utf-8"))
            len_entity = len(self.entities[sample_index].decode("utf-8"))

            # for word matching
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

            for i in range(len_mention):
                for j in range(len_entity):
                    # for sentence matching
                    self.tensor[sample_index][3][i][j] = 1 - spatial.distance.cosine(
                        sentence_embedding_m[sample_index][i], sentence_embedding_e[sample_index][j])

                    # for string matching
                    if self.mentions[sample_index][i] == self.entities[sample_index][j]:
                        self.tensor[sample_index][0][i][j] = 1

                    # for character matching
                    character_embedding_mi = self.get_embedding(self.mentions[sample_index][i],
                                                                self.character_embedding_model)
                    character_embedding_ei = self.get_embedding(self.entities[sample_index][j],
                                                                self.character_embedding_model)
                    self.tensor[sample_index][1][i][j] = 1 - spatial.distance.cosine(character_embedding_ei,
                                                                                     character_embedding_mi)

        print("Tensor Completed!")

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


    def getIDFWeights(self, x_names, x_index, vocab_id_w, idfModel):
        res = []  # idf_weights
        length = len(x_names)

        for i in range(length):
            name = x_names[i]
            index = x_index[i]
            character_idfs = dict()
            default_idf = 1  # for unknown word, assign idf=1

            name_segs = list(jieba.cut(name))

            for name in name_segs:
                for character in name:
                    if name in idfModel.keys():
                        character_idfs[character] = idfModel[name]
                    else:
                        character_idfs[character] = default_idf

            weight_per_name = []
            for character_index in index:
                character = vocab_id_w[character_index]
                if character in character_idfs.keys():
                    weight_per_name.append(character_idfs[character])
                else:
                    weight_per_name.append(default_idf)
            res.append(weight_per_name)
            res_arr = np.asarray(res)

            # *** normalize the idf weights
            row_sums = res_arr.sum(axis=1)
            res_arr = res_arr / row_sums[:, np.newaxis]
        return res_arr


    def getAttention(self, r, x, index, vocab_id_w, idfModel):
        # r(input_size, None, hidden_n * 2) => (None, input_size, hidden_n * 2)
        representation = np.transpose(r, (1, 0, 2))

        weights = self.getIDFWeights(x, index, vocab_id_w, idfModel)  # shape: batch_size * sequence_length

        # weights transform from 2D to 3 D and then 3D*3D broadcasting
        representation = representation * weights[:, :, np.newaxis]
        return representation


    def getAttention_M(self, r, m, x, index, vocab_id_w, idfModel):
        # r(input_size, None, hidden_n * 2) => (None, input_size, hidden_n * 2)
        representation = np.transpose(r, (1, 0, 2))
        representation = representation * m[:, :, np.newaxis]

        weights = self.getIDFWeights(x, index, vocab_id_w, idfModel)  # shape: batch_size * sequence_length

        # weights transform from 2D to 3 D and then 3D*3D broadcasting
        representation = representation * weights[:, :, np.newaxis]
        return representation
    def no_attention(self, r, m, x, index, vocab_id_w, idfModel):
        # r(input_size, None, hidden_n * 2) => (None, input_size, hidden_n * 2)
        representation = np.transpose(r, (1, 0, 2))
        representation = representation * m[:, :, np.newaxis]

        # weights = self.getIDFWeights(x, index, vocab_id_w, idfModel)  # shape: batch_size * sequence_length

        # weights transform from 2D to 3 D and then 3D*3D broadcasting
        # representation = representation * weights[:, :, np.newaxis]
        return representation

    def getSentence_Embedding(self, x1, x2, max_document_length):
        checkpoint_dir = os.path.abspath(os.path.join(self.bilstm_dir, "checkpoints"))
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        checkpoint_file = ckpt.model_checkpoint_path

        vocab_file = os.path.join(checkpoint_dir, "vocab")

        inpH = InputHelper()
        vocab_processor = MyVocabularyProcessor(max_document_length, min_frequency=0)
        vocab_processor = vocab_processor.restore(vocab_file)

        tmp = []
        (x1_index, x2_index, mask_x1, mask_x2, tmp) = inpH.get_data(vocab_processor, x1, x2, tmp,
                                                                    max_document_length)
        # x1_index, x2_index = inpH.toVocabularyIndexVector(x1, x2, vocab_file, max_document_length)

        idfModel = loadIDFModel(self.idfModel_file)

        # # load vocabulary model
        # vocab_processor = MyVocabularyProcessor(max_document_length, min_frequency=0)
        # vocab_processor = vocab_processor.restore(vocab_file)

        # Extract word:id mapping from the object.
        vocab_dict = vocab_processor.vocabulary_._mapping
        vocab_id_w = dict((y, x) for x, y in vocab_dict.iteritems())

        print("\nGenerating Sentence Embedding Result...\n")
        graph = tf.Graph()

        with graph.as_default():
            sess = tf.Session()

            with sess.as_default():
                # Load the saved meta graph and restore variables
                saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
                sess.run(tf.initialize_all_variables())
                saver.restore(sess, checkpoint_file)

                # Get the placeholders from the graph by name
                # the output is a list with only one element
                input_x1 = graph.get_operation_by_name("input_x1").outputs[0]
                input_x2 = graph.get_operation_by_name("input_x2").outputs[0]

                sentence_representation1 = graph.get_operation_by_name("sentence_embedding/Representation1").outputs[0]
                sentence_representation2 = graph.get_operation_by_name("sentence_embedding/Representation2").outputs[0]
                print "Sentence vector shape after sentence modeling"
                print sentence_representation2.get_shape().as_list()

                r1, r2 = sess.run([sentence_representation1, sentence_representation2],
                                  {input_x1: x1_index, input_x2: x2_index})

                # Applied Attention_mechanism
                representation1 = self.getAttention_M(r1, mask_x1, x1, x1_index, vocab_id_w, idfModel)
                representation2 = self.getAttention_M(r2, mask_x2, x2, x2_index, vocab_id_w, idfModel)
                # representation1 = self.no_attention(r1, mask_x1, x1, x1_index, vocab_id_w, idfModel)
                # representation2 = self.no_attention(r2, mask_x2, x2, x2_index, vocab_id_w, idfModel)

        return representation1, representation2
