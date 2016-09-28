#matplotlib inline
from __future__ import print_function
import collections
import math
import numpy as np
import os
import random
import tensorflow as tf
import zipfile
import operator
from six.moves import range
import string
import sys
from tensorflow.python.framework import tensor_shape
from matplotlib import pylab
from six.moves.urllib.request import urlretrieve
from sklearn.manifold import TSNE

filename = 'text8.zip'

def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        for name in f.namelist():
            return tf.compat.as_str(f.read(name))
        f.close

text = read_data(filename)
print('Data size: %d' % len(text))

te_size = 1000
va_size = 1000
te_text = text[:te_size]
va_text = text[te_size: (va_size + te_size)]
tr_text = text[(va_size + te_size):]
tr_size = len(tr_text)
print(tr_size, tr_text[:64])
print(va_size, va_text[:64])

# '#' - PAD, '@' - GO, '!' - EOS
vocab = ['#', '@', '!', '.', ' '] + list(string.ascii_lowercase)
vocab_size = len(vocab)

id_to_c = {i:c for i,c in enumerate(vocab)}
c_to_id = {id_to_c[k]:k for k in id_to_c}

"""
Generate batch_size number of sequences of varying lengths
such that lengths of sequences are between max_len/2 - max_len
"""

def gen_batch(text, batch_size, max_len = 128):
    lens = np.random.choice(range(max_len // 2, max_len), batch_size, True)
    #start, len pairs
    pairs = [(np.random.randint(0, len(text) - max_len + 1), l) for l in lens]
    batch = [text[i:(i+l)] + '.' + '#'*(max_len - l -1)  for i,l in pairs]
    return batch

def gen_pad_labels(batch, max_len):
    labels = []
    weights = []
    for b in batch:
        text = b.split('.') # separate '.' and trailing #s
        words = text[0].split(' ')
        rev = '@' + ' '.join([w[::-1] for w in words]) + '.' + '!'
        w = [1.0]*(len(rev) - 1) + [0.0]*(max_len - len(rev))
        rev = rev + '#'*(max_len - len(rev))
        labels.append(rev)
        weights.append(w)
    return labels, weights

def rev_batch(batch):
    return [b[::-1] for b in batch]

def one_hot(l):
    h = np.zeros((len(l), vocab_size))
    h[np.arange(len(l)), l] = 1.0
    return h

def batch_to_train(batch):
    batch = [[c_to_id[c] for c in b] for b in batch]
    batch = np.asarray(batch).T.tolist()
    batch = [one_hot(b) for b in batch]
    return batch

def get_batch(text, size, in_len, out_len):
    b = gen_batch(text, size, in_len)
    l, w = gen_pad_labels(b, out_len)
    w = np.asarray(reduce(operator.add, zip(*w)))
    return batch_to_train(rev_batch(b)), batch_to_train(l), w

def tr_batch_to_str(batch, rev = False):
    batch = [[id_to_c[c] for c in np.argmax(s, 1)] for s in batch]
    if rev == True:
        batch = [''.join(s)[::-1] for s in np.asarray(batch).T.tolist()]
    else:
        batch = [''.join(s) for s in np.asarray(batch).T.tolist()]
    return batch

def perp(predictions, labels):
    predictions[predictions < 1e-10] = 1e-10
    return np.exp(np.sum(np.multiply(labels, -np.log(predictions)))
            / labels.shape[0])

n_nodes = 512
#buckets = [[16, 18], [32,34], [64, 66], [128, 130]]
buckets = [[4,6], [8,10], [16, 18], [32, 34]]

graph = tf.Graph()
with graph.as_default():
    # Parameters:
    # encoder weights
    e_iw = tf.Variable(tf.truncated_normal([vocab_size, 4*n_nodes], -0.1, 0.1))
    e_ow = tf.Variable(tf.truncated_normal([n_nodes, 4*n_nodes], -0.1, 0.1))
    e_cb = tf.Variable(tf.zeros([1, 4 * n_nodes]))

    # decoder weights
    d_iw = tf.Variable(tf.truncated_normal([vocab_size, 4*n_nodes], -0.1, 0.1))
    d_ow = tf.Variable(tf.truncated_normal([n_nodes, 4*n_nodes], -0.1, 0.1))
    d_cb = tf.Variable(tf.zeros([1, 4 * n_nodes]))
    
    # Classifier weights and biases.
    d_w = tf.Variable(tf.truncated_normal([n_nodes, vocab_size], -0.1, 0.1))
    d_b = tf.Variable(tf.zeros([vocab_size]))

    # Definition of the cell computation.
    def lstm_cell(i, o, state, iw, ow, cb):
        """Create a LSTM cell. See e.g.: http://arxiv.org/pdf/1402.1128v1.pdf
           Note that in this formulation, we omit the various connections
           between the previous state and the gates."""
        cal = tf.matmul(i, iw) + tf.matmul(o, ow) + cb

        input_gate = tf.sigmoid(cal[:,0:n_nodes])
        forget_gate = tf.sigmoid(cal[:,n_nodes:2*n_nodes])
        update = cal[:, 2*n_nodes:3*n_nodes]
        state = forget_gate * state + input_gate * tf.tanh(update)
        output_gate = tf.sigmoid(cal[:,3*n_nodes:])
        return output_gate * tf.tanh(state), state

    def encoder_lstm(i, o, state):
        return lstm_cell(i, o, state, e_iw, e_ow, e_cb)

    def decoder_lstm(i, o, state):
        return lstm_cell(i, o, state, d_iw, d_ow, d_cb)

    def encode(inputs):
        batch_size = tf.shape(inputs[0])[0]
        shape = [batch_size] + tensor_shape.as_shape(n_nodes).as_list() #UGLY
        output = tf.zeros(tf.pack(shape))
        state = tf.zeros(tf.pack(shape))

        for inp in inputs:
            inp = tf.nn.dropout(inp, 0.9)
            output, state = encoder_lstm(inp, output, state)
        return state

    def get_seq(e_state, go, output_len):
        outputs = list()
        output = tf.zeros_like(e_state)
        state = e_state
        pred = []

        inp = go
        for i in range(output_len):
            output, state = decoder_lstm(inp, output, state)
            outputs.append(output)
            # sample output to get next input
            logits = calc_logits(output)
            inp = sample_argmax_char(logits)
            pred.append(tf.nn.softmax(logits))
        return pred

    def decode(e_state, inputs):
        batch_size = tf.shape(inputs[0])[0]
        shape = [batch_size] + tensor_shape.as_shape(n_nodes).as_list() #UGLY

        outputs = list()
        output = tf.zeros(tf.pack(shape))
        state = e_state

        for i in inputs: # decoder training
            i = tf.nn.dropout(i, 0.9)
            output, state = decoder_lstm(i, output, state)
            outputs.append(tf.nn.dropout(output, 0.9))
        return outputs

    def calc_loss(logits, labels):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                  logits, labels) * target_weights)
        return loss

    def sample_argmax_char(predictions):
        h = tf.one_hot(tf.argmax(predictions, 1), vocab_size,
                dtype=tf.float32)
        return h

    def calc_logits(outputs):
        # Classifier.
        h = tf.matmul(outputs, d_w) + d_b
        return h

    # Input data.
    e_inputs = [tf.placeholder(tf.float32, shape=[None, vocab_size])
            for i in range(buckets[-1][0])]
    d_inputs = [tf.placeholder(tf.float32, shape=[None, vocab_size])
            for i in range(buckets[-1][1])]
    target_weights = tf.placeholder(tf.float32)
 
    seqs = []
    losses = []
    opts = []
    # Optimizer.
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(
            10.0, global_step, 3000, 0.5, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    for b in buckets:
        e_input = e_inputs[:b[0]]
        d_input = d_inputs[0:(b[1]-1)]
        d_labels = d_inputs[1:b[1]]
        state = encode(e_input)
        output = decode(state, d_input)
        logits = calc_logits(tf.concat(0, output))
        loss = calc_loss(logits, tf.concat(0, d_labels))
        gradients, v = zip(*optimizer.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
        opt = optimizer.apply_gradients(
            zip(gradients, v), global_step=global_step)
        seqs.append(get_seq(state, d_input[0], b[1] - 1))
        losses.append(loss)
        opts.append(opt)


tr_batch_size = 64
va_batch_size = 16
te_batch_size = 4

va_e_seqs, va_d_seqs, _ = get_batch(va_text, va_batch_size, buckets[-1][0],
        buckets[-1][1])
te_e_seqs, te_d_seqs, _ = get_batch(te_text, te_batch_size, buckets[-1][0],
        buckets[-1][1])

num_steps = 14001
summ_freq = 100

def model_step(session, e_seqs, d_seqs, b, pred_only = False, tw = None):
    input_dict = {}
    input_dict.update({e_inputs[i]:e_seqs[i] for i in range(len(e_seqs))})
    input_dict.update({d_inputs[i]:d_seqs[i] for i in range(len(d_seqs))})
    if tw is not None:
        input_dict[target_weights] = tw

    if not pred_only:
        to_fetch = [opts[b], losses[b], learning_rate]
        _, l, lr = session.run(to_fetch, input_dict)
        return l, lr, None
    else:
        seq = session.run([seqs[b]], input_dict)
        return None, None, seq[0]

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print('Initialized')
    print('Nodes:', n_nodes)
    print('Buckets:', buckets)
    mean_loss = 0
    bucket = 0
    # prepare training data
    samples_per_bucket = tr_batch_size * 500
    tr_data = []
    for b_id in range(len(buckets)):
        b = buckets[b_id]
        for i in range(samples_per_bucket // tr_batch_size):
            e_seqs, d_seqs, w = get_batch(tr_text, tr_batch_size, b[0], b[1])
            tr_data.append([e_seqs, d_seqs, w, b_id])

    for step in range(num_steps):
        index = step % len(tr_data)
        [e_seqs, d_seqs, w, b] = tr_data[index]
        l, lr, _ = model_step(session, e_seqs, d_seqs, b, tw = w)
        mean_loss += l
        if step % summ_freq == 0:
            mean_loss = mean_loss / summ_freq if step > 0 else mean_loss
            print('Avg loss at step %d: %f lr: %f' % (step, mean_loss, lr))
            mean_loss = 0
            print('Mini perp: %.2f' % np.float(np.exp(l)))

            # Measure validation set perplexity.
            vb = len(buckets) - 1
            _, _, seq = model_step(session, va_e_seqs, va_d_seqs, vb, True)
            print('Val perp: %.2f' % perp(np.concatenate(seq),
                np.concatenate(va_d_seqs[1:])))

            # Run on test samples
            if step % (summ_freq * 10) == 0:
                _, _, seq = model_step(session, te_e_seqs, te_d_seqs,
                        len(buckets) - 1, True)
                labels = np.concatenate(te_d_seqs[1:])
                # Print some samples.
                print('=' * 80)
                print('Test perp: %.2f' % perp(np.concatenate(seq)
                    , labels))
                # convert to one-hot for printing, sample using argmax
                seq = [one_hot(np.argmax(b, 1)) for b in seq]
                
                input_seq = tr_batch_to_str(te_e_seqs, True)
                exp_seq = tr_batch_to_str(te_d_seqs[1:])
                pred_seq = tr_batch_to_str(seq)
                for i in range(te_batch_size):
                    print('I:', input_seq[i])
                    print('E:', exp_seq[i])
                    print('O:', pred_seq[i])
                    print('--')
                print('=' * 80)
