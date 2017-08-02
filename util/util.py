#!/usr/bin/env python
# coding=utf-8
import re
import numpy as np


def write_evaluation_file(eval_file, right_output_file, wrong_output_file, labels, predictions, mentions, entities):
    tp, fp, tn, fn = 0, 0, 0, 0
    length = len(predictions)
    right_output_file.write("mention\tentity\tprediction\tlabel\n")
    wrong_output_file.write("mention\tentity\tprediction\tlabel\n")

    for i in range(length):
        str_m = mentions[i] + "\t" + entities[i] + "\t" + str(predictions[i]) + "\t" + str(labels[i]) + "\n"
        if predictions[i] == 1 and labels[i] == 1:
            tp += 1.0
            right_output_file.write(str_m)
        elif predictions[i] == 1 and labels[i] == 0:
            fn += 1.0
            wrong_output_file.write(str_m)
        elif predictions[i] == 0 and labels[i] == 1:
            fp += 1.0
            wrong_output_file.write(str_m)
        else:
            tn += 1.0
            right_output_file.write(str_m)

    eval_file.write("True positive: " + str(tp) + "\n")
    eval_file.write("False positive: " + str(fp) + "\n")
    eval_file.write("True negative: " + str(tn) + "\n")
    eval_file.write("False negative: " + str(fn) + "\n")

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (recall + precision)

    eval_file.write("Precision:" + str(precision) + "\n")
    eval_file.write("Recall:" + str(recall) + "\n")
    eval_file.write("F1:" + str(f1) + "\n\n")


def write_evaluation_file_multi(eval_file, right_output_file, wrong_output_file, labels, predictions, labels2,
                                predictions2, mentions1, entities1, mentions2, entities2):
    tp, fp, tn, fn = 0, 0, 0, 0
    length = len(predictions)
    right_output_file.write("mention_description\tentity_description\tprediction_description\tlabel"
                            "\tmention_operation\tentity_operation\tprediction_operation\tlabel \n")
    wrong_output_file.write("mention_description\tentity_description\tprediction_description\tlabel"
                            "\tmention_operation\tentity_operation\tprediction_operation\tlabel \n")

    for i in range(length):
        str_m = mentions1[i] + "\t" + entities1[i] + "\t" + str(predictions[i]) + "\t" + str(labels[i]) + "\t" + \
                mentions2[i] + "\t" + entities2[i] + "\t" + str(predictions2[i]) + "\t" + str(labels2[i]) + "\n"
        if predictions[i] == 1 and labels[i] == 1:
            tp += 1.0
            right_output_file.write(str_m)
        elif predictions[i] == 1 and labels[i] == 0:
            fn += 1.0
            wrong_output_file.write(str_m)
        elif predictions[i] == 0 and labels[i] == 1:
            fp += 1.0
            wrong_output_file.write(str_m)
        else:
            tn += 1.0
            right_output_file.write(str_m)

        eval_file.write("True positive: " + str(tp) + "\n")
        eval_file.write("False positive: " + str(fp) + "\n")
        eval_file.write("True negative: " + str(tn) + "\n")
        eval_file.write("False negative: " + str(fn) + "\n")

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (recall + precision)

        eval_file.write("Precision:" + str(precision) + "\n")
        eval_file.write("Recall:" + str(recall) + "\n")
        eval_file.write("F1:" + str(f1) + "\n")


def preprocess_unit(str):
    res_0 = re.sub(ur"\w+", '', str)
    res_0 = re.sub(ur"[-（ ）\( \)， \.；;、：° \s+ \*\[ \] \+ ？? \,]", '', res_0)

    # res_0 = str.replace('&nbsp;', '')
    # res_0 = re.sub(ur"\u3000", '', res_0)  # 将中文的空格用英文空格代替，后面可以处理

    # res_0 = re.sub(ur"\[?[（]?\w+[.]?\w+\]?[）]?$", '', res_0)  # 去除掉ICD编码 eg:I20.222
    #
    # res_0 = re.sub(r"\w\d+.\d+", '', res_0)  # 去除掉ICD编码 eg:I20.222
    # res_0 = re.sub(r"\w\d+.?x+\d+$", '', res_0)  # 去除掉尾部的编码 eg:I20.x222
    #
    # res_0 = re.sub(r"\s\w+", "", res_0)  # 去掉空格后的字母，eg: 心肌梗塞急性 NOS
    # res_0 = re.sub(ur"\[\w+\]", "", res_0).strip()  # 去掉括号中的字母解释，eg: [NSSMD]
    # res_0 = re.sub(ur"（\w+）", "", res_0).strip()  # 去掉括号中的字母解释，eg: （NSSMD）
    # res_0 = re.sub(ur"\(\w+\)", "", res_0).strip()  # 去掉括号中的字母解释，eg: (NSSMD)
    #
    # res = re.split(ur"[（ ）\( \)， \.；;、：° \s+ \*\[ \] \+ ？? \,]", res_0)
    # res = filter(lambda x: len(x) != 1 and len(x) != 0, res)
    #
    # return "".join(res)
    return res_0


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

    # add unknown symbol
    vocab.append("<UNK>")
    embd.append([0.0] * 100)

    vocab_size = len(vocab)
    embedding_dim = len(embd[0])
    embedding = np.asarray(embd)
    return vocab, vocab_size, embedding_dim, embedding
