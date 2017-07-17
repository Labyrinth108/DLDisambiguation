#!/usr/bin/env python
# coding=utf-8
import re
import numpy as np


def preprocess_unit(str):
    res_0 = str.replace('&nbsp;', '')
    res_0 = re.sub(ur"\u3000", '', res_0)  # 将中文的空格用英文空格代替，后面可以处理
    res_0 = re.sub(ur"\[?[（]?\w+[.]?\w+\]?[）]?$", '', res_0)  # 去除掉ICD编码 eg:I20.222

    res_0 = re.sub(r"\w\d+.\d+", '', res_0)  # 去除掉ICD编码 eg:I20.222
    res_0 = re.sub(r"\w\d+.?x+\d+$", '', res_0)  # 去除掉尾部的编码 eg:I20.x222

    res_0 = re.sub(r"\s\w+", "", res_0)  # 去掉空格后的字母，eg: 心肌梗塞急性 NOS
    res_0 = re.sub(ur"\[\w+\]", "", res_0).strip()  # 去掉括号中的字母解释，eg: [NSSMD]
    res_0 = re.sub(ur"（\w+）", "", res_0).strip()  # 去掉括号中的字母解释，eg: （NSSMD）
    res_0 = re.sub(ur"\(\w+\)", "", res_0).strip()  # 去掉括号中的字母解释，eg: (NSSMD)

    res = re.split(ur"[（ ）\( \)， \.；;、：° \s+ \*\[ \] \+ ？? \,]", res_0)

    res = filter(lambda x: len(x) != 1 and len(x) != 0, res)

    return "".join(res)


def preprocess_arr(arr):
    res = []
    for i in arr:
        res.append(preprocess_unit(i.decode("utf-8")))
    return res


def loadIDFModel(file_path):
    file = open(file_path)
    idfModel = dict()
    for line in file.readlines():
        segs = line.strip().split(":")
        name, idf = segs[0].decode("utf-8"), float(segs[1])
        idfModel[name] = idf
    return idfModel


def load_data(path):
    file = open(path)
    line = file.readline()
    res = []
    y = []

    while line != "":
        tmp = line.split(",")
        res.append([tmp[0], tmp[1]])
        y.append(tmp[2])
        line = file.readline()
    file.close()
    return res, y


def loadWord2Vec(filename):
    vocab = []
    embd = []
    file = open(filename, 'r')
    line = file.readline()
    while line != "":
        line = file.readline()  # jump the first line
        if line == "":
            break
        row = line.strip().split(' ')
        vocab.append(row[0])
        embedding_float = [float(i) for i in row[1:]]
        embd.append(embedding_float)
    print('Loaded Word2vec model!')
    file.close()
    return vocab, embd


def getEmbedding(filename):
    vocab, embd = loadWord2Vec(filename)
    vocab.append("<UNK>")
    embd.append([0.0] * 100)
    vocab_size = len(vocab)
    embedding_dim = len(embd[0])
    embedding = np.asarray(embd)
    return vocab, vocab_size, embedding_dim, embedding
