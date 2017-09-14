# coding:utf-8
import numpy as np
from matplotlib.font_manager import FontManager, FontProperties
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
from util.input_helpers import InputHelper

max_document_length = 20
task_num = 2
name = "des" if task_num == 1 else "opr"

arr = np.load(
    "/Users/luoyi/Documents/Python/DLDisambiguation/Tensor_files/0823/Length" + str(
        max_document_length) + "/test_" + name + ".npy")

inpH = InputHelper()

data_file_test = "/Users/luoyi/Documents/Python/DLDisambiguation/data/test_data_0823_" + name + ".txt"
x_test_mention, x_test_entity, y_test = inpH.getTsvTestData(data_file_test, "\t", max_document_length, y_value=False)

sample_n = 1
dir_ = "./fig/"


# sample_n = len(arr)

def getChineseFont():
    return FontProperties(fname='/Users/luoyi/Downloads/msyh.ttf')


sample_index = 352
row_n, col_n = 1, 4
plt.subplots(row_n, col_n, figsize=(20, 10))

for j in range(col_n):
    t = arr[sample_index, :, :, j]
    # x = t
    # t = (x - np.min(x)) / (np.max(x) - np.min(x))
    # t = np.abs(t)
    plt.subplot(row_n, col_n, j + 1)
    plt.imshow(t, vmin=-1, vmax=1, interpolation='nearest', cmap=plt.cm.magma)
    # plt.imshow(t, vmin=-1, vmax=1, interpolation='nearest', cmap=plt.cm.bone)
    # plt.imshow(t, vmin=-1, vmax=1, interpolation='nearest', cmap=plt.cm.hsv)
    # plt.imshow(t, vmin=-1, vmax=1, interpolation='nearest', cmap=plt.cm.prism)
    # plt.imshow(t, vmin=-1, vmax=1, interpolation='nearest', cmap=plt.cm.flag)
    # plt.imshow(t, vmin=-1, vmax=1, interpolation='nearest', cmap=plt.cm.autumn)
    # plt.imshow(t, vmin=-1, vmax=1, interpolation='nearest', cmap=plt.cm.cool)
    # plt.imshow(t, vmin=-1, vmax=1, interpolation='nearest', cmap=plt.cm.jet)
    # plt.imshow(t, vmin=-1, vmax=1, interpolation='nearest', cmap=plt.get_cmap('gray'))

    plt.title(str(j))

# plt.ylabel(unicode(x_test_mention[sample_index]), fontproperties=getChineseFont())
# plt.xlabel(unicode(x_test_entity[sample_index]), fontproperties=getChineseFont())

# plt.show()
plt.colorbar(fraction=0.046, pad=0.04)
plt.savefig(dir_ + "outfile" + str(sample_index) + name + "_magma.jpg")
