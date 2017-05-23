import numpy as np
from tensor import Tensor

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
    return res, y

def get_Data(path):
    data, y = load_data(path)
    res = [] # num * (m*n)

    for pair in data:
        t = Tensor(pair[0], pair[1])
        res.append(t.get_tensor())
    return res, y