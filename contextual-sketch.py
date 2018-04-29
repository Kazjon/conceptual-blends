"""Training script for conditional seq2seq sketch model."""
from __future__ import division

import os
from random import shuffle, randint

import numpy as np
import tensorflow as tf

from rnn_model import rnn_model

N_CLASSES = 345


def rand_batch_gen(dataset):
    """Old batch generation."""
    while True:  # use try catch here; just repeat idx=.. and batch=...
        idx = randint(0, len(dataset)-1)  # choose a random batch id
        ys_ = np.array([randint(0, N_CLASSES - 1)])  # output
        br_ = ys_
        yield dataset[idx][np.newaxis, :], ys_, br_


def pad_sequences(sequences):
    """Saves sequences in a numpy array adding zeros if they are too short."""
    max_len = max(s.shape[0] for s in sequences)
    padded = []
    for seq in sequences:
        zero_pad = np.concatenate(
            [seq, np.zeros((max_len - seq.shape[0], ) + seq.shape[1:])])
        padded.append(zero_pad[np.newaxis, :])

    return np.concatenate(padded, axis=0)


def gen_batches(data, batch_size=8, randomize=False):
    """Creates batches in the format expacted by the RNN """
    indices = list(range(len(data)))
    if randomize:
        shuffle(indices)

    for start in range(0, len(data), batch_size):
        yield pad_sequences(data[indices[start:start + batch_size]])


def split_dataset(samples, ratio=0.8):
    """Split dataset randomly in train and test set."""
    nsamples = len(samples)
    num_train = int(ratio*nsamples)

    # shuffle samples
    shuffle(samples)

    trainset = samples[:num_train]
    testset = samples[num_train:]

    return trainset, testset


def build_lstm_cell(size, keep_prob=None):
    """define lstm cell for encoder"""
    encoder_cell = tf.contrib.rnn.LSTMCell(size)
    if keep_prob is not None:
        encoder_cell = tf.contrib.rnn.DropoutWrapper(
            encoder_cell, output_keep_prob=keep_prob)

    return encoder_cell


def code_stroke_tags(predictions):
    """Converts the last feature dimension to 0 or 1 based on sign."""
    # condition = predictions[:, :, -1:] > 0
    condition = False  # set all to 0
    stroke_tags = tf.where(condition,
                           tf.ones_like(predictions[:, :, -1:]),
                           tf.zeros_like(predictions[:, :, -1:]))

    return tf.concat([predictions[:, :, :-1], stroke_tags], axis=2)


def decode_state(cells, weights, b_o):
    """Converts the LSTM state to a sketch coordinate."""
    return tf.tensordot(cells, weights, axes=1) + b_o


def map_to_scope(var_list):
    """Loads the given scoped variables from an unscoped checkpoint."""
    return {var.op.name.split('/', 1)[1]: var for var in var_list}


def load_classification_parameters(sess, ckpt_file, scope):
    """Load the variables of the classification model."""
    class_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
    class_saver = tf.train.Saver(var_list=map_to_scope(class_vars))
    class_saver.restore(sess, ckpt_file)


class contextual_seq2seq(object):

    def __graph__(self, state_size, vocab_size, num_layers, ext_context_size):
        """build network graph"""
        tf.reset_default_graph()

        # placeholders
        xs_ = tf.placeholder(dtype=tf.float32,
                             shape=[None, None, vocab_size], name='xs')
        ys_ = tf.placeholder(dtype=tf.int32, shape=[None, ],
                             name='ys')  # decoder targets
        dec_inputs_length_ = tf.placeholder(dtype=tf.int32, shape=[None, ],
                                            name='dec_inputs_length')
        ext_context_ = tf.placeholder(dtype=tf.int32, shape=[None, ],
                                      name='ext_context')

        # dropout probability (shared)
        keep_prob_ = tf.placeholder(tf.float32)

        # stack cells
        encoder_cell = tf.contrib.rnn.MultiRNNCell(
            [build_lstm_cell(size, keep_prob_)
             for size in [state_size]*num_layers],
            state_is_tuple=True)

        with tf.variable_scope('encoder') as scope:
            # define encoder
            _, enc_context = tf.nn.dynamic_rnn(
                cell=encoder_cell, dtype=tf.float32, inputs=xs_)

        with tf.variable_scope('proj') as scope:
            # output projection
            V = tf.get_variable(
                'V', shape=[state_size, vocab_size],
                initializer=tf.contrib.layers.xavier_initializer())
            b_o = tf.get_variable('bo', shape=[vocab_size, ],
                                  initializer=tf.constant_initializer(0.))

            # class embedding
            class_proj = tf.get_variable(
                'C', shape=[N_CLASSES, state_size],
                initializer=tf.contrib.layers.xavier_initializer())
            c_o = tf.get_variable('co', shape=[state_size, ],
                                  initializer=tf.constant_initializer(0.))

        context_proj = tf.one_hot(ext_context_, N_CLASSES,
                                  on_value=1.0, off_value=0.0)
        context_proj = tf.tensordot(context_proj, class_proj, axes=1) + c_o

        # embedding for pad symbol
        pad = tf.zeros(shape=(tf.shape(xs_)[0], vocab_size), dtype=tf.float32)

        def loop_fn_initial(time, cell_output, cell_state, loop_state):
            """init function for raw_rnn"""
            assert cell_output is None
            assert loop_state is None
            assert cell_state is None

            return ((time >= dec_inputs_length_),  # elements_finished
                    pad,                           # initial_input
                    enc_context,                   # initial_cell_state
                    None,                          # emit_output
                    None)                          # initial_loop_state

        def loop_fn(time, cell_output, cell_state, loop_state):
            """state transition function for raw_rnn"""
            if cell_state is None:    # time == 0
                return loop_fn_initial(
                    time, cell_output, cell_state, loop_state)

            emit_output = cell_output  # == None for time == 0

            # couple external context with cell states (c, h)
            next_cell_state = []
            for layer in range(num_layers):
                next_cell_state.append(tf.contrib.rnn.LSTMStateTuple(
                    c=cell_state[layer].c + context_proj,
                    h=cell_state[layer].h + context_proj))

            next_cell_state = tuple(next_cell_state)

            elements_finished = (time >= dec_inputs_length_)
            finished = tf.reduce_all(elements_finished)

            # TODO: the current code computes an intermediate generated
            #   sequence for conditioning purposes: is it what we want?
            out = decode_state(cell_output, V, b_o)
            out.set_shape([None, vocab_size])
            next_input = tf.cond(finished, lambda: pad, lambda: out)

            next_loop_state = None

            return (elements_finished,
                    next_input,
                    next_cell_state,
                    emit_output,
                    next_loop_state)

        # define the decoder with raw_rnn <- loop_fn, loop_fn_initial
        decoder_cell = tf.contrib.rnn.MultiRNNCell(
            [build_lstm_cell(size) for size in [state_size]*num_layers],
            state_is_tuple=True)

        with tf.variable_scope('decoder') as scope:
            decoder_outputs_ta, _, _ = tf.nn.raw_rnn(decoder_cell, loop_fn)
            decoder_outputs = tf.transpose(decoder_outputs_ta.stack(),
                                           perm=[1, 0, 2])

        # decode final states
        added_strokes = code_stroke_tags(decode_state(decoder_outputs, V, b_o))

        # generate a whole new sequence
        predictions = added_strokes
        # .. or add the generated strokes
        # predictions = tf.concat([xs_, added_strokes], axis=1)

        with tf.variable_scope('class') as scope:
            logits = rnn_model(predictions, num_classes=N_CLASSES)

        # optimization
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                labels=ys_)
        loss = tf.reduce_mean(losses)

        # train only encoder / decoder
        tvar = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "encoder")
        tvar += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "decoder")
        tvar += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "proj")

        train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(
            loss, var_list=tvar)

        # attach symbols to class
        self.loss = loss
        self.train_op = train_op
        self.predictions = predictions
        self.ext_context_size = ext_context_size
        self.keep_prob_ = keep_prob_  # placeholders
        self.xs_ = xs_
        self.ys_ = ys_
        self.dec_inputs_length_ = dec_inputs_length_
        self.ext_context_ = ext_context_

    def __init__(self, state_size, vocab_size, num_layers, ext_context_size,
                 model_name='contextual_seq2seq',
                 ckpt_path='ckpt/contextual_seq2seq/'):

        self.model_name = model_name
        self.ckpt_path = ckpt_path

        if not os.path.isdir(ckpt_path):
            os.makedirs(ckpt_path)

        # build graph
        self.__graph__(state_size, vocab_size, num_layers, ext_context_size)

    def train(self, trainset, testset, niter=1000, ntest=300, epochs=int(1e6)):
        """Train using niter iterations per epoch and for the given number of
        epochs."""

        print('\n>> Training begins!\n')

        def fetch_dict(datagen, keep_prob=0.5):
            """Format model input data."""
            bx, by, br = next(datagen)
            while not (bx.shape[0] > 0 and bx.shape[1] > 0):
                bx, by, br = next(datagen)

            dec_lengths = np.full((bx.shape[0], ), bx.shape[1], dtype=np.int32)

            feed_dict = {
                self.xs_: bx,
                self.ys_: by,
                self.dec_inputs_length_: dec_lengths,
                self.ext_context_: br,
                self.keep_prob_: keep_prob
            }
            return feed_dict

        # setup session
        saver = tf.train.Saver()
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        load_classification_parameters(
            sess, "models/model.ckpt-1500000", "class")

        # get last checkpoint
        ckpt = tf.train.get_checkpoint_state(self.ckpt_path)
        # verify it
        if ckpt and ckpt.model_checkpoint_path:
            try:
                saver.restore(sess, ckpt.model_checkpoint_path)
            except tf.OpError:
                # graph structure changed, cannot load, restart training
                pass

        try:
            # start training
            for j in range(epochs):
                mean_loss = 0
                for n_it in range(niter):
                    _, loss = sess.run(
                        [self.train_op, self.loss],
                        feed_dict=fetch_dict(trainset))
                    mean_loss += loss
                    print('  [{}/{}]\r'.format(n_it, niter))

                print('[{}] train loss : {}'.format(j, mean_loss / niter))
                saver.save(sess, self.ckpt_path + self.model_name + '.ckpt',
                           global_step=j)

                # evaluate
                testloss = 0
                for _ in range(ntest):
                    testloss += sess.run(
                        [self.loss],
                        feed_dict=fetch_dict(testset, keep_prob=1.))[0]
                print('test loss : {}'.format(testloss / ntest))

        except KeyboardInterrupt:
            print('\n>> Interrupted by user at iteration {}'.format(j))


def main():
    """Script entry point"""
    np.random.seed(42)  # ensure deterministic results

    with np.load('rainbow_data.npz', encoding='latin1') as fid:
        # split into batches
        data_ = fid['arr_0']

    # split data
    traindata, testdata = split_dataset(data_)

    # prepare train set generator
    # generators (iterators) everytime they give a new sample
    trainset = rand_batch_gen(traindata)
    testset = rand_batch_gen(testdata)

    # create a model
    model = contextual_seq2seq(state_size=1024, vocab_size=3,
                               num_layers=3, ext_context_size=1)
    # train
    model.train(trainset, testset, niter=10, ntest=3, epochs=100)


if __name__ == '__main__':
    main()
