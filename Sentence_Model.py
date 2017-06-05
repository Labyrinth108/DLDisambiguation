import os

import jieba
import numpy as np
import tensorflow as tf

from DLDisambiguation.util.input_helpers import InputHelper
from DLDisambiguation.util.preprocess import MyVocabularyProcessor
from DLDisambiguation.util.util import loadIDFModel


def getIDFWeights(x_names, x_index, vocab_id_w, idfModel):
    res = []  # idf_weights
    length = len(x_names)

    for i in range(length):
        name = x_names[i]
        index = x_index[i]
        character_idfs = dict()
        default_idf = 1  # for unknown word, assign idf=1

        name_segs = list(jieba.cut(name))

        for name in name_segs:
            for character in name:
                if name in idfModel.keys():
                    character_idfs[character] = idfModel[name]
                else:
                    character_idfs[character] = default_idf

        weight_per_name = []
        for character_index in index:
            character = vocab_id_w[character_index]
            if character in character_idfs.keys():
                weight_per_name.append(character_idfs[character])
            else:
                weight_per_name.append(default_idf)
        res.append(weight_per_name)
        res_arr = np.asarray(res)

        # normalize the idf weights
        row_sums = res_arr.sum(axis=1)
        res_arr = res_arr / row_sums[:, np.newaxis]
    return res_arr


def getAttention(r, x, index, vocab_id_w, idfModel):
    representation = np.transpose(r, (1, 0, 2))
    weights = getIDFWeights(x, index, vocab_id_w, idfModel)  # shape: batch_size * sequence_length
    representation = representation * weights[:, :, np.newaxis]  # 2D*3D
    return representation


def getSentence_Embedding(x1, x2):
    max_document_length = 15
    # timestamp = "1496560933"  # sequence_length = 15
    timestamp = "1496582776"  # sequence_length = 20
    dir = "/Users/luoyi/Documents/Python/DLDisambiguation"

    out_dir = os.path.abspath(os.path.join(dir, "Bi-LSTM/runs", timestamp))
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    checkpoint_file = ckpt.model_checkpoint_path

    vocab_file = os.path.join(checkpoint_dir, "vocab")

    inpH = InputHelper()
    x1_index, x2_index = inpH.toVocabularyIndexVector(x1, x2, vocab_file, max_document_length)

    idfModel_file = os.path.join(dir, "data/idfModel.txt")
    idfModel = loadIDFModel(idfModel_file)

    # load vocabulary model
    vocab_processor = MyVocabularyProcessor(max_document_length, min_frequency=0)
    vocab_processor = vocab_processor.restore(vocab_file)

    # Extract word:id mapping from the object.
    vocab_dict = vocab_processor.vocabulary_._mapping
    vocab_id_w = dict((y, x) for x, y in vocab_dict.iteritems())

    print("\nGenerating Sentence Embedding Result...\n")
    graph = tf.Graph()

    with graph.as_default():
        sess = tf.Session()

        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            sess.run(tf.initialize_all_variables())
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_x1 = graph.get_operation_by_name("input_x1").outputs[0]
            input_x2 = graph.get_operation_by_name("input_x2").outputs[0]

            sentence_representation1 = graph.get_operation_by_name("sentence_embedding/Representation1").outputs[0]
            sentence_representation2 = graph.get_operation_by_name("sentence_embedding/Representation2").outputs[0]

            r1, r2 = sess.run([sentence_representation1, sentence_representation2],
                              {input_x1: x1_index, input_x2: x2_index})
            # Applied Attention_mechanism
            representation1 = getAttention(r1, x1, x1_index, vocab_id_w, idfModel)
            representation2 = getAttention(r2, x2, x2_index, vocab_id_w, idfModel)

    return representation1, representation2
