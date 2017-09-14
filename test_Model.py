import tensorflow as tf
import os
from DLDisambiguation.util.input_helpers import InputHelper
from tensor import Tensor
import numpy as np
from util.util import write_evaluation_file

tf.flags.DEFINE_string("train_dir", "./", "Training dir root")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

task_num = 2
inpH = InputHelper()
max_document_length = 20

name = "des" if task_num == 1 else "opr"

# load in model
model_dir = "./runs/Single_task21501595265"
checkpoint_dir = os.path.join(model_dir, "checkpoints")
print(checkpoint_dir)
ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
checkpoint_file = ckpt.model_checkpoint_path

lstm_dir = "Description1500991322" if task_num == 1 else "Operation1501000120"
lstm_dir = os.path.join("./Sentence_Modeling/runs", lstm_dir)

# load data
load_Tensor = True

data_file = os.path.join(FLAGS.train_dir, "data/training_data_0724_" + name + ".txt")
data_file_test = os.path.join(FLAGS.train_dir, "data/test_data_0724_" + name + ".txt")
data_file_val = os.path.join(FLAGS.train_dir, "data/validation_data_0724_" + name + ".txt")

x_test_mention, x_test_entity, y_test = inpH.getTsvTestData(data_file_test, "\t", max_document_length, y_value=False)

if load_Tensor:
    mydir = "./Length" + str(max_document_length) + "/"
    x_test_tensor = np.load(mydir + "test_" + name + ".npy")

else:
    x_test_tensor = Tensor(x_test_mention, x_test_entity, len(x_test_entity), max_document_length, task_num,
                           lstm_dir).get_tensor()
    x_test_tensor = x_test_tensor.transpose((0, 2, 3, 1))

graph = tf.Graph()

eval_file = open(os.path.join(model_dir, "test_analysis.txt"), "w+")
rightfile = open(os.path.join(model_dir, "right_analysis.txt"), "w+")
wrongfile = open(os.path.join(model_dir, "wrong_analysis.txt"), "w+")

with graph.as_default():
    sess = tf.Session()

    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        sess.run(tf.initialize_all_variables())
        saver.restore(sess, checkpoint_file)

        for var in tf.trainable_variables():
            print(var.name)
            print(var)
            print(sess.run(var))

        # Get the placeholders from the graph by name
        # the output is a list with only one element
        input_tensor = graph.get_operation_by_name("input_tensor").outputs[0]
        input_y = graph.get_operation_by_name("input_y").outputs[0]
        droppout = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        predictions = graph.get_operation_by_name("output/predictions").outputs[0]
        accuracy = graph.get_operation_by_name("accuracy/accuracy").outputs[0]
        scores = graph.get_operation_by_name("output/scores").outputs[0]

        pre, accu, scores = sess.run(
            [predictions, accuracy, scores], {input_tensor: x_test_tensor, input_y: y_test, droppout: 1})

        eval_file.write("Accu: " + str(accu) + "\n")
        length = len(x_test_mention)
        labels = np.argmax(y_test, 1)

        write_evaluation_file(eval_file, rightfile, wrongfile, labels, pre, x_test_mention, x_test_entity)
