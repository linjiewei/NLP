import tensorlayer as tl
import tensorflow as tf
from loadvec import load_vectors
import time
import numpy as np
from datetime import datetime

top_k_list = [1, 3, 5, 10]
print_length = 100

model_file_name = "model_generate_text.npz"

dictionary, reverse_dictionary, embedding_vec = load_vectors('data/4_fasttext_model.vec')

init_scale = 0.1
learning_rate = 1.0
max_grad_norm = 5
sequence_length = 70
embedding_size = 300
hidden_size = 250
max_epoch = 20
max_max_epoch = 400
lr_decay = 0.95
batch_size = 10
vocabulary_size = len(dictionary)


def clear_unknow_words(words, dictionary):
    return list(filter(lambda i: i in dictionary, words))


with open('data/3_Tianlongbabu_segmented.txt', 'r', encoding='utf8') as f:
    words = f.read().split()
    words = clear_unknow_words(words, dictionary)
    words_id = tl.nlp.words_to_word_ids(words, dictionary)

input_data = tf.placeholder(tf.int32, [batch_size, sequence_length])
targets = tf.placeholder(tf.int32, [batch_size, sequence_length])

input_data_test = tf.placeholder(tf.int32, [1, 1])


def inference(x, is_train, sequence_length, reuse=None):
    rnn_init = tf.random_uniform_initializer(-init_scale, init_scale)
    embed_tensor = tf.constant_initializer(embedding_vec)
    with tf.variable_scope('model', reuse=reuse):
        network = tl.layers.EmbeddingInputlayer(
            inputs=x,
            embedding_size=300,
            vocabulary_size=vocabulary_size,
            E_init=embed_tensor,
        )
        network = tl.layers.BiRNNLayer(
            network,
            cell_fn=tf.contrib.rnn.BasicLSTMCell,
            cell_init_args={
                'forget_bias': 0.0,
                'state_is_tuple': True
            },
            n_hidden=hidden_size,
            initializer=rnn_init,
            n_steps=sequence_length,
            return_last=False,
            return_seq_2d=True,
            name='lstm1'
        )
        lstm1 = network
        network = tl.layers.DenseLayer(
            network,
            n_units=vocabulary_size,
            W_init=rnn_init,
            b_init=rnn_init,
            act=tf.identity
        )
    return network, lstm1


network, lstm1 = inference(input_data, is_train=True, sequence_length=sequence_length, reuse=None)
network_test, lstm1_test = inference(input_data_test, is_train=False, sequence_length=1, reuse=True)

y_linear = network_test.outputs
y_soft = tf.nn.softmax(y_linear)

with tf.variable_scope('learning_rate'):
    lr = tf.Variable(0.0, trainable=False)

sess = tf.InteractiveSession()
tl.layers.initialize_global_variables(sess)

load_params = tl.files.load_and_assign_npz(sess, name='./models/2018-05-06-12-47-ep300.npz', network=network)

for top_k in top_k_list:
    seed = "段譽 猛地 使劍".split()
    state1_fw = tl.layers.initialize_rnn_state(lstm1_test.fw_initial_state)
    state1_bw = tl.layers.initialize_rnn_state(lstm1_test.bw_initial_state)
    outs_id = tl.nlp.words_to_word_ids(seed, dictionary)
    for ids in outs_id[:-1]:
        a_id = np.asarray(ids).reshape(1, 1)
        state1_fw, state1_bw = sess.run([lstm1_test.fw_final_state, lstm1_test.bw_final_state], feed_dict={
            input_data_test: a_id,
            lstm1_test.fw_initial_state: state1_fw,
            lstm1_test.bw_initial_state: state1_bw,
        })

    a_id = outs_id[-1]
    for _ in range(print_length):
        a_id = np.asarray(a_id).reshape(1, 1)
        out, state1_fw, state1_bw = sess.run([y_soft, lstm1_test.fw_final_state, lstm1_test.bw_final_state], feed_dict={
            input_data_test: a_id,
            lstm1_test.fw_initial_state: state1_fw,
            lstm1_test.bw_initial_state: state1_bw,
        })
        a_id = tl.nlp.sample_top(out[0], top_k=top_k)
        outs_id.append(a_id)
    sentense_words = tl.nlp.word_ids_to_words(outs_id, reverse_dictionary)
    sentense = ''.join(sentense_words)
    print(f'{top_k}:  {sentense}')
