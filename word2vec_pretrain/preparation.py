#!/usr/bin/env python
# coding=utf-8

from sklearn.feature_extraction.text import TfidfVectorizer
import jieba
import re
import gensim
import codecs
import os
from DLDisambiguation.util.util import preprocess_unit

def generateProcessedCorpusFile(src_file_path, character_file_path, word_file_path):
    """
    载入语料，去掉NULL的无效数据，
    分词用空格隔开每个词并生成word_src.txt
    将每个字用空格隔开并生成character_src.txt
    :param src_file_path:
    :param character_file_path:
    :param word_file_path:
    :return:
    """
    data = []
    file = codecs.open(src_file_path, "r")
    line = file.readline()

    character_file = codecs.open(character_file_path, "w+", "utf-8")
    word_file = codecs.open(word_file_path, "w+", "utf-8")

    while line != "":
        d = line.strip().decode("utf-8")
        # 处理"无"，"NA"."NULL".""的情况
        if len(d) != 1 and d != "NA" and d != "NULL" and d != "":
            data.append(d)
        line = file.readline()

    for i in data:
        res = preprocess_unit(i)
        characters = list("".join(res))
        if len(characters) == 0:
            continue
        character_file.write(" ".join(characters) + "\n")
    character_file.close()
    print("Finished character model!")

    for i in data:
        res = preprocess_unit(i)
        x = jieba.cut(res)
        words = list(x)
        if len(words) == 0:
            continue
        word_file.write(" ".join(words) + "\n")
    word_file.close()
    print("Finished Word model!")


def get_IDF(corpus_file, idf_file):
    file = open(corpus_file)
    line = file.readline()
    counter = 0
    corpus = []

    while line != "":
        counter += 1

        line = preprocess_unit(line)
        cut_words = list(jieba.cut(line))
        cut_words_valid = [i for i in cut_words if re.sub("\w+", "", i) != ""]
        content = " ".join(cut_words_valid)
        corpus.append(content)
        line = file.readline()
    file.close()

    # compute idf model
    vectorizer = TfidfVectorizer(
        use_idf=True,
        norm=None,
        smooth_idf=False,  # idf = ln(N+1 / ni+1)
        sublinear_tf=False,  # tf = 1+ln(tf)
        binary=False,
        min_df=1, max_df=1.0, max_features=None,
        strip_accents='unicode',
        ngram_range=(1, 1), preprocessor=None, stop_words=None, tokenizer=None, vocabulary=None
    )
    X = vectorizer.fit_transform(corpus)
    idf = vectorizer.idf_
    name_idf_dict = dict(zip(vectorizer.get_feature_names(), idf))

    # write IDF_Model
    idf_file = open(idf_file, "w+")
    for k, v in name_idf_dict.items():
        idf_file.write(k.encode("utf-8") + ":" + str(v) + "\n")
    idf_file.close()
    print("IDF_Model Finished!")


class MySentences(object):
    def __init__(self, fname):
        self.fname = fname

    def __iter__(self):
        for line in open(self.fname):
            yield line.split()


def checkPath(x):
    if not os.path.exists(x):
        open(x, 'a').close()


def compute_word2vec(embedding_dir, type):
    load_model_flag = False
    src_file = embedding_dir + type + "_src.txt"
    model_file = embedding_dir + type + '.model'
    wv_file = embedding_dir + type + 'character_model.txt'

    if not load_model_flag:
        sentences = MySentences(src_file)  # a memory-friendly iterator
        model = gensim.models.Word2Vec(sentences)

        checkPath(model_file)
        checkPath(wv_file)

        model.save(model_file)  # save model
        model.wv.save_word2vec_format(wv_file, binary=False)  # save word2vec txt
        print("Word2vec model finished!\n")
    else:
        model = gensim.models.Word2Vec.load(model_file)  # load in model


if __name__ == "__main__":

    corpus_file = "../data/d2013_operation.txt"
    idf_model = "../data/idfModel_operation.txt"

    # corpus_file = "../data/db_description_d2013.txt"
    # idf_model = "../data/idfModel.txt"

    task_num = 1
    if task_num == 1:
        embedding_dir = "../data/word2vec_new/"
    else:
        embedding_dir = "../data/operation/"

    # generate IDF Model
    get_IDF(corpus_file, idf_model)

    # produce character_src_file and word_src_file from corpus file
    # generateProcessedCorpusFile(corpus_file, character_src_file, word_src_file)

    # generate word2vec for characters and words
    # compute_word2vec(embedding_dir, "character")
    # compute_word2vec(embedding_dir, "word")