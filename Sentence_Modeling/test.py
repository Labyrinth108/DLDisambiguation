from DLDisambiguation.util.input_helpers import InputHelper
from DLDisambiguation.util.preprocess import MyVocabularyProcessor
import tensorflow as tf
import os

# tf.flags.DEFINE_string("test_file", "../data/validation_data_0724_opr.txt", "training file (default: None)")
tf.flags.DEFINE_string("test_file", "../data/test_data_0724_des.txt", "training file (default: None)")

FLAGS = tf.flags.FLAGS
inpH = InputHelper()
max_document_length = 20
y_is_value = True

model_dir = "./runs/Description1501174251"
# model_dir = "./runs/Description1500991322"  # 0.760
# model_dir = "./runs/1500428748" # 0.69
# model_dir = "./runs/Description1500983617"  # 0.767
# model_dir = "./runs/Description1501058401" # 0.754
# model_dir = "./runs/Operation1501000120" # 0.809

checkpoint_dir = os.path.join(model_dir, "checkpoints")
print(checkpoint_dir)
ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
checkpoint_file = ckpt.model_checkpoint_path

vocab_file = os.path.join(checkpoint_dir, "vocab")
# load vocabulary model
vocab_processor = MyVocabularyProcessor(max_document_length, min_frequency=0)
vocab_processor = vocab_processor.restore(vocab_file)

test_x1, test_x2, test_y = inpH.getTestIndexedDataSet(FLAGS.test_file, "\t", vocab_processor, max_document_length,
                                                      y_is_value)

graph = tf.Graph()

with graph.as_default():
    sess = tf.Session()
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        sess.run(tf.initialize_all_variables())
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        # the output is a list with only one element
        input_x1 = graph.get_operation_by_name("input_x1").outputs[0]
        input_x2 = graph.get_operation_by_name("input_x2").outputs[0]
        input_y = graph.get_operation_by_name("input_y").outputs[0]

        accuracy_o = graph.get_operation_by_name("accuracy/accuracy").outputs[0]

        accu = sess.run([accuracy_o], {input_x1: test_x1, input_x2: test_x2, input_y: test_y})
    print(accu)
