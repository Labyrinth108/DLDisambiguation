import tensorflow as tf
import os
from DLDisambiguation.util.input_helpers import InputHelper
import matplotlib.pyplot as plt
import numpy as np


def plot_conv(sess, t_vars, name):
    var_conv = [v for v in t_vars if name in v.name]
    W = var_conv[0]  # [2, 2, 1, 8]
    W = sess.run(W)

    length = W.shape[-1]

    row_n = 2 if length == 8 else 4
    col_n = length / row_n
    plt.subplots(row_n, col_n)

    for i in range(length):
        axes = plt.subplot(row_n, col_n, i + 1)
        map = W[:, :, 0, i]
        plt.imshow(map,cmap=plt.cm.magma)
        # plt.imshow(map, cmap=plt.get_cmap('gray'))
        # plt.xlabel(i)
        axes.set_xticks([])
        axes.set_yticks([])
    # plt.colorbar(fraction=0.046, pad=0.04)
    plt.savefig(dir_ + "map" + name + ".jpg")


def plot_activation(sample_index, k, row_n, col_n, conv, name):
    for i in range(col_n):
        k += 1
        m = plt.subplot(row_n, col_n, k)
        if i == col_n / 2:
            m.set_title(name)
        # plt.imshow(conv[sample_index, :, :, i], vmin=-1, vmax=1, interpolation='nearest', cmap=plt.cm.jet)
        plt.imshow(conv[sample_index, :, :, i], interpolation='nearest', cmap=plt.cm.magma)
        # plt.imshow(conv[sample_index, :, :, i], cmap=plt.cm.jet, aspect='auto')
        # plt.imshow(conv[sample_index, :, :, i], cmap=plt.get_cmap('gray'), aspect='auto')
        m.set_xticks([])
        m.set_yticks([])
    # plt.title(name)
    # plt.colorbar(fraction=0.046, pad=0.04)
    return k


def plot_activations(conv1, conv2, conv3, conv4):
    length = conv1.shape[-1]
    row_n = 4
    col_n = length
    plt.subplots(row_n, col_n, figsize=(20, 10))
    sample_idnex = 352

    k = 0
    k = plot_activation(sample_idnex, k, row_n, col_n, conv1, "Str")
    k = plot_activation(sample_idnex, k, row_n, col_n, conv2, "Character Embedding")
    k = plot_activation(sample_idnex, k, row_n, col_n, conv3, "Word Embedding")
    plot_activation(sample_idnex, k, row_n, col_n, conv4, "Sentence Embedding")

    # plt.colorbar()
    plt.savefig(dir_ + str(sample_idnex) + "activations_color" + ".jpg")


tf.flags.DEFINE_string("train_dir", "./", "Training dir root")
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

task_num = 1
inpH = InputHelper()
max_document_length = 10
dir_ = "fig/"

model_dir = "./runs/NewExp/Single_task11503543419"
checkpoint_dir = os.path.join(model_dir, "checkpoints")
print(checkpoint_dir)
ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
checkpoint_file = ckpt.model_checkpoint_path
#
# mydir = "./Length" + str(max_document_length) + "/"
# x_test_tensor = np.load(mydir + "test_des" + ".npy")
x_test_tensor = np.load("./Tensor_files/0823/Length10/test_des.npy")
graph = tf.Graph()

with graph.as_default():
    sess = tf.Session()

    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        sess.run(tf.initialize_all_variables())
        saver.restore(sess, checkpoint_file)

        # PLOT Conv Filters
        t_vars = tf.trainable_variables()
        for var in t_vars:
            print(var.name)
            print(var)
            print(sess.run(var))
        plot_conv(sess, t_vars, "conv1")
        plot_conv(sess, t_vars, "conv1_1")
        plot_conv(sess, t_vars, "conv1_2")
        plot_conv(sess, t_vars, "conv1_3")

        input_tensor = graph.get_operation_by_name("input_tensor").outputs[0]
        droppout = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        conv1 = graph.get_operation_by_name("conv1/conv1").outputs[0]
        conv2 = graph.get_operation_by_name("conv1_1/conv1").outputs[0]
        conv3 = graph.get_operation_by_name("conv1_2/conv1").outputs[0]
        conv4 = graph.get_operation_by_name("conv1_3/conv1").outputs[0]

        conv_layer1, conv_layer2, conv_layer3, conv_layer4 = sess.run([conv1, conv2, conv3, conv4],
                                                                      feed_dict={input_tensor: x_test_tensor})
        plot_activations(conv_layer1, conv_layer2, conv_layer3, conv_layer4)
