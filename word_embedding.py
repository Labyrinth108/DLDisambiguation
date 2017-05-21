#!/usr/bin/env python
# coding=utf-8

import gensim

class MySentences(object):
    def __init__(self, fname):
        self.fname = fname

    def __iter__(self):
        for line in open(self.fname):
            yield line.split()

dir = "/Users/luoyi/Documents/Python/DLDisambiguation/data/"
character_src_file = dir + "character_src.txt"
character_model_file = dir + 'character.model'
character_wv_file = dir + 'character_model.txt'

word_src_file = dir + "word_src.txt"
word_model_file = dir + 'word.model'
word_wv_file = dir + 'word_model.txt'

load_model_flag = False

if not load_model_flag:
    sentences = MySentences(character_src_file) # a memory-friendly iterator
    model = gensim.models.Word2Vec(sentences)

    model.save(character_model_file) # save model
    model.wv.save_word2vec_format(character_wv_file, binary=False)
else:
    model = gensim.models.Word2Vec.load(character_model_file) # load in model

if not load_model_flag:
    sentences = MySentences(word_src_file) # a memory-friendly iterator
    model = gensim.models.Word2Vec(sentences)

    model.save(word_model_file) # save model
    model.wv.save_word2vec_format(word_wv_file, binary=False)
else:
    model = gensim.models.Word2Vec.load(word_model_file) # load in model

# str_l = list("不稳定性心绞痛".decode("utf-8"))
# for c in str_l:
#     try:
#         m = c.encode("utf-8")
#         print(model.wv[m])
#     except Exception as e:
#         print(c)
#         print(str_l.index(c))
