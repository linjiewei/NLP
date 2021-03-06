import tensorlayer as tl
import tensorflow as tf
from loadvec import load_vectors
import time
import numpy as np
from datetime import datetime

top_k_list = [1, 3, 5, 10]
print_length = 200

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
        network = tl.layers.RNNLayer(
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
            return_seq_2d=False,
            name='lstm1'
        )
        lstm1 = network
        network = tl.layers.RNNLayer(
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
            name='lstm2'
        )
        lstm2 = network
        network = tl.layers.DenseLayer(
            network,
            n_units=vocabulary_size,
            W_init=rnn_init,
            b_init=rnn_init,
            act=tf.identity
        )
    return network, lstm1, lstm2


def loss_fn(outputs, targets, batch_size, sequence_length):
    loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
        [outputs],
        [tf.reshape(targets, [-1])],
        [tf.ones([batch_size * sequence_length])]
    )
    cost = tf.reduce_sum(loss) / batch_size
    return cost


network, lstm1, lstm2 = inference(input_data, is_train=True, sequence_length=sequence_length, reuse=None)
network_test, lstm1_test, lstm2_test = inference(input_data_test, is_train=False, sequence_length=1, reuse=True)

y_linear = network_test.outputs
y_soft = tf.nn.softmax(y_linear)

cost = loss_fn(network.outputs, targets, batch_size, sequence_length)

with tf.variable_scope('learning_rate'):
    lr = tf.Variable(0.0, trainable=False)

tvars = network.all_params[1:]
grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), max_grad_norm)
optimizer = tf.train.GradientDescentOptimizer(lr)
train_op = optimizer.apply_gradients(zip(grads, tvars))

sess = tf.InteractiveSession()
tl.layers.initialize_global_variables(sess)

train_data = words_id

for i in range(max_max_epoch):
    new_lr_decay = lr_decay ** max(i - max_epoch, 0.0)
    sess.run(tf.assign(lr, learning_rate * new_lr_decay))
    print(f'Epoch: {i+1}/{max_max_epoch} Learning rate: {sess.run(lr)}')

    epoch_size = ((len(train_data) // batch_size) - 1) // sequence_length

    start_time = time.time()

    costs = 0.0
    iters = 0
    state1 = tl.layers.initialize_rnn_state(lstm1.initial_state)
    state2 = tl.layers.initialize_rnn_state(lstm2.initial_state)
    for step, (x, y) in enumerate(tl.iterate.ptb_iterator(train_data, batch_size, sequence_length)):
        _cost, state1, state2, _ = sess.run([cost, lstm1.final_state, lstm2.final_state, train_op], feed_dict={
            input_data: x,
            targets: y,
            lstm1.initial_state: state1,
            lstm2.initial_state: state2,
        })
        costs += _cost
        iters += 1
        if step % (epoch_size // 10) == 1:
            print("%.3f perplexity: %.3f speed: %.0f wps" %
                  (step * 1.0 / epoch_size,
                   np.exp(costs / iters),
                   iters * batch_size / (time.time() - start_time)
                   ))

    if i % 100 == 0 and i!=0:
        now = datetime.now()
        tl.files.save_npz(network.all_params, f'./models/{now.strftime("%Y-%m-%d-%H-%M")}-ep{i}', sess)

    for top_k in top_k_list:
        seed = "那 西夏".split()
        state1 = tl.layers.initialize_rnn_state(lstm1_test.initial_state)
        state2 = tl.layers.initialize_rnn_state(lstm2_test.initial_state)
        outs_id = tl.nlp.words_to_word_ids(seed, dictionary)
        for ids in outs_id[:-1]:
            a_id = np.asarray(ids).reshape(1, 1)
            state1, state2 = sess.run([lstm1_test.final_state, lstm2_test.final_state], feed_dict={
                input_data_test: a_id,
                lstm1_test.initial_state: state1,
                lstm2_test.initial_state: state2
            })

        a_id = outs_id[-1]
        for _ in range(print_length):
            a_id = np.asarray(a_id).reshape(1, 1)
            out, state1, state2 = sess.run([y_soft, lstm1_test.final_state, lstm2_test.final_state], feed_dict={
                input_data_test: a_id,
                lstm1_test.initial_state: state1,
                lstm2_test.initial_state: state2
            })
            a_id = tl.nlp.sample_top(out[0], top_k=top_k)
            outs_id.append(a_id)
        sentense_words = tl.nlp.word_ids_to_words(outs_id, reverse_dictionary)
        sentense = ''.join(sentense_words)
        print(f'{top_k}:  {sentense}')
