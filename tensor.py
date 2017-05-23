import numpy as np
import  gensim, jieba

class Tensor:

    # m--mention str
    # e--entity str
    def __init__(self, m, e):
        self.mention = m
        self.entity = e
        self.mention_length = len(m)
        self.entity_length = len(e)

        dir = "/Users/luoyi/Documents/Python/DLDisambiguation/data/"
        character_model_file = dir + 'character.model'
        word_mode_file = dir + 'word.model'
        self.character_embedding_model = gensim.models.Word2Vec.load(character_model_file)
        self.word_embedding_model = gensim.models.Word2Vec.load(word_mode_file)

        self.tensor = np.zeros(shape=(self.mention_length, self.entity_length, 4))
        self.init_str_matrix()
        self.init_character_matrix()
        self.init_word_matrix()
        self.init_sentence_matrix()

    def get_tensor(self):
        return self.tensor

    # initialize the str-match matrix
    def init_str_matrix(self):
        matrix = self.tensor[0]
        for i in range(self.mention_length):
            for j in range(self.entity_length):
                if self.mention[i] == self.entity[j]:
                    matrix[i,j] = 1

    def get_embedding(self, word, model):
        if word in model.wv.vocab:
            return model.wv[word]
        else:
            vector_length = 100
            return np.zeros([vector_length])

    # initialize the character-match matrix
    def init_character_matrix(self):
        matrix = self.tensor[1]

        for i in range(self.mention_length):
            for j in range(self.entity_length):
                character_embedding_mi = self.get_embedding(self.mention[i], self.character_embedding_model)
                character_embedding_ei = self.get_embedding(self.entity[i], self.character_embedding_model)
                matrix[i, j] = character_embedding_ei * character_embedding_mi

    # initialize the word-match matrix
    def init_word_matrix(self):
        matrix = self.tensor[2]
        words_m = list(jieba.cut(self.mention))
        words_e = list(jieba.cut(self.entity))
        len_w_m = len(words_m)
        len_w_e = len(words_e)

        for i in range(len_w_m):
            for j in range(len_w_e):
                words_sim = self.get_embedding(words_m[i]) * self.get_embedding(words_e[j])

                # assign the word_pair_sim to the character_pairs which construct the words
                for character_i in words_m[i]:
                    for character_j in words_e[j]:
                        matrix[self.mention.index(character_i), self.entity.index(character_j)] = words_sim

    def get_sentence_embedding(self):


    #  initialize the sentence_level-match matrix
    def init_sentence_matrix(self):
        matrix = self.tensor[3]
        sentence_embedding_m = self.get_sentence_embedding(self.mention)
        sentence_embedding_e = self.get_sentence_embedding(self.entity)

        for i in range(self.mention_length):
            for j in range(self.entity_length):
                matrix[i, j] = sentence_embedding_m[i] * sentence_embedding_e[j]
