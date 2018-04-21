import tensorflow as tf
import numpy as np

from random import sample, shuffle, randint

import sys


def rand_batch_gen(dataset):
    n_class = 345
    while True:  # use try catch here; just repeat idx=.. and batch=...
        idx = randint(0, len(dataset)-1)  # choose a random batch id
        ys_ = np.zeros(n_class)  # output
        ys_[randint(0, n_class)] = 1
        br = ys_
        yield dataset[idx][np.newaxis, :], ys_, br


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


def split_dataset(samples, ratio=[0.8, 0.2]):

    nsamples = len(samples)
    num_train = int(ratio[0]*nsamples)

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


class contextual_seq2seq(object):

    def __init__(self, state_size, vocab_size, num_layers,
                 ext_context_size,
                 model_name='contextual_seq2seq',
                 ckpt_path='ckpt/contextual_seq2seq/'):

        self.model_name = model_name
        self.ckpt_path = ckpt_path

        def __graph__():
            """build network graph"""
            tf.reset_default_graph()

            # placeholders
            xs_ = tf.placeholder(dtype=tf.float32,
                                 shape=[None, None, vocab_size], name='xs')
            ys_ = tf.placeholder(dtype=tf.float32, shape=[None, ],
                                 name='ys')  # decoder targets
            dec_inputs_length_ = tf.placeholder(dtype=tf.int32, shape=[None, ],
                                                name='dec_inputs_length')
            ext_context_ = tf.placeholder(dtype=tf.float32, shape=[None, ],
                                          name='ext_context')

            # dropout probability (shared)
            keep_prob_ = tf.placeholder(tf.float32)

            # stack cells
            sizes = [vocab_size] + [state_size]*(num_layers - 1)
            encoder_cell = tf.contrib.rnn.MultiRNNCell(
                [build_lstm_cell(size, keep_prob_) for size in sizes],
                state_is_tuple=True)

            with tf.variable_scope('encoder') as scope:
                # define encoder
                enc_op, enc_context = tf.nn.dynamic_rnn(
                    cell=encoder_cell, dtype=tf.float32, inputs=xs_)

            # output projection
            V = tf.get_variable(
                'V', shape=[state_size, vocab_size],
                initializer=tf.contrib.layers.xavier_initializer())
            bo = tf.get_variable('bo', shape=[vocab_size],
                                 initializer=tf.constant_initializer(0.))

            ###
            # embedding for pad symbol
            pad = tf.zeros(shape=(tf.shape(xs_)[0], ), dtype=tf.float32)

            ###
            # init function for raw_rnn
            def loop_fn_initial(time, cell_output, cell_state, loop_state):
                assert cell_output is None
                assert loop_state is None
                assert cell_state is None

                elements_finished = (time >= dec_inputs_length_)
                initial_input = pad
                initial_cell_state = enc_context
                initial_loop_state = None

                return (elements_finished,
                        initial_input,
                        initial_cell_state,
                        None,
                        initial_loop_state)

            ###
            # state transition function for raw_rnn
            def loop_fn(time, cell_output, cell_state, loop_state):

                if cell_state is None:    # time == 0
                    return loop_fn_initial(
                        time, cell_output, cell_state, loop_state)

                emit_output = cell_output  # == None for time == 0
                #
                # couple external context with cell states (c, h)
                next_cell_state = []
                for layer in range(num_layers):
                    next_cell_state.append(tf.contrib.rnn.LSTMStateTuple(
                        c=cell_state[layer].c + ext_context_,
                        h=cell_state[layer].h + ext_context_))

                next_cell_state = tuple(next_cell_state)

                elements_finished = (time >= dec_inputs_length_)
                finished = tf.reduce_all(elements_finished)

                def search_for_next_input():
                    output = tf.matmul(cell_output, V) + bo
                    return tf.argmax(output, axis=1)
                
                next_input = tf.cond(finished,
                                     lambda: pad, search_for_next_input)

                next_loop_state = None

                return (elements_finished,
                        next_input,
                        next_cell_state,
                        emit_output,
                        next_loop_state)

            ###
            # define the decoder with raw_rnn <- loop_fn, loop_fn_initial
            decoder_cell = tf.contrib.rnn.MultiRNNCell(
                [build_lstm_cell(state_size) for _ in range(num_layers)],
                state_is_tuple=True)

            with tf.variable_scope('decoder') as scope:
                decoder_outputs_ta, _, _ = tf.nn.raw_rnn(decoder_cell, loop_fn)
                decoder_outputs = decoder_outputs_ta.stack()

            ####
            # flatten states to 2d matrix for matmult with V
            dec_op_reshaped = tf.reshape(decoder_outputs, [-1, state_size])
            # /\_o^o_/\
            logits = tf.matmul(dec_op_reshaped, V) + bo
            #
            # predictions
            predictions = tf.nn.softmax(logits)
            #
            # optimization
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=tf.reshape(ys_, [-1]))
            loss = tf.reduce_mean(losses)
            train_op = tf.train.AdagradOptimizer(
                learning_rate=0.1).minimize(loss)
            #
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
            #####
        ####
        # build graph
        __graph__()

    def train(self, trainset, testset, n, epochs):

        print('\n>> Training begins!\n')

        def fetch_dict(datagen, keep_prob=0.5):
            while not (bx.shape[0] > 0 and bx.shape[1] > 0):
                bx, by, br = next(datagen)

            dec_lengths = np.full((bx.shape[0], ), by.shape[0], dtype=np.int32)

            feed_dict = {
                self.xs_: bx,
                self.ys_: by,
                self.dec_inputs_length_: dec_lengths,
                self.ext_context_: br,
                self.keep_prob_: keep_prob
            }
            return feed_dict

        ##
        # setup session
        saver = tf.train.Saver()
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        # get last checkpoint
        ckpt = tf.train.get_checkpoint_state(self.ckpt_path)
        # verify it
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        try:
            # start training
            for j in range(epochs):
                mean_loss = 0
                for i in range(n):
                    _, l = sess.run([self.train_op, self.loss],
                                    feed_dict=fetch_dict(trainset))
                    mean_loss += l
                    sys.stdout.write('[{}/{}]\r'.format(i, n))

                print('\n>> [{}] train loss at : {}'.format(j, mean_loss/n))
                saver.save(sess, self.ckpt_path + self.model_name + '.ckpt',
                           global_step=i)
                #
                # evaluate
                testloss = 0
                for k in range(300):
                    testloss += sess.run(
                        [self.loss],
                        feed_dict=fetch_dict(testset, keep_prob=1.))[0]
                print('test loss : {}'.format(testloss / 300))

        except KeyboardInterrupt:
            print('\n>> Interrupted by user at iteration {}'.format(j))


def main():
    """Script entry point"""
    # gather data
    with np.load('rainbow_data.npz', encoding='latin1') as fid:
        # split into batches
        data_ = fid['arr_0']

    # split data
    traindata, testdata = split_dataset(data_)

    # prepare train set generator
    # generators (iterators) everytime they give a new sample
    trainset = rand_batch_gen(traindata)
    testset = rand_batch_gen(testdata)

    ###
    # infer vocab size
    # vocab_size = len(metadata['idx2w'])
    # ext_context_size = metadata['respect_size']
    #
    # create a model
    model = contextual_seq2seq(state_size=1024, vocab_size=3,
                               num_layers=3, ext_context_size=1)
    # train
    model.train(trainset, testset, n=1000, epochs=1000000)


if __name__ == '__main__':
    main()
