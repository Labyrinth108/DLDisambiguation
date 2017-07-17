#!/usr/bin/env python
# coding=utf-8

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['AR PL KaitiM GB']})

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from gensim.models import Word2Vec

def checkSimilarity(model_file, word):

    model = Word2Vec.load(model_file)
    arr = model.most_similar([word])
    for x in arr:
        print(str(x[0] + ":" + str(x[1])))

def main():
    # model_file = "../data/word2vec/character.model"
    model_file = "../data/word2vec_new/word.model"
    checkSimilarity(model_file, "左")

    # character_wv_file = '../data/word2vec/character_model.txt'
    # word_wv_file = '../data/word2vec/word_model.txt'
    #
    # embeddings_file = word_wv_file
    # wv, vocabulary = load_embeddings(embeddings_file)
    #
    # tsne = TSNE(n_components=2, random_state=0)
    # np.set_printoptions(suppress=True)
    # Y = tsne.fit_transform(wv[:1000, :])
    #
    # plt.scatter(Y[:, 0], Y[:, 1])
    # for label, x, y in zip(vocabulary, Y[:, 0], Y[:, 1]):
    #     plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    # plt.show()


def load_embeddings(file_name):

    # with codecs.open(file_name, 'r', 'utf-8') as f_in:
    lines = open(file_name).readlines()[1:]
    vocabulary, wv = zip(*[line.strip().split(' ', 1) for line in lines])

    wv = np.loadtxt(wv)
    return wv, vocabulary

if __name__ == '__main__':
    main()