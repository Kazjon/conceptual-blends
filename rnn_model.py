"""CNN-RNN classification model."""
import tensorflow as tf


class Params(dict):
    """Access dictionary elements like object members."""

    def __init__(self, *args, **kwargs):
        super(Params, self).__init__(*args, **kwargs)
        self.__dict__ = self


N_CLASSES = 345

DEFAULT_PARAMS = Params({
    'batch_size': 2,
    'dropout': 0.3,
    'batch_norm': False,
    'num_conv': [48, 64, 96],
    'conv_len': [5, 5, 3],
    'num_layers': 3,
    'num_nodes': 128})


def _sequence_length(sequence):
    """Compute sequence lengths on a batch."""
    used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
    return tf.cast(tf.reduce_sum(used, 1), tf.int32)


def _add_conv_layers(features, params, index, is_training=False):
    """Adds convolution layers."""
    if params.batch_norm:
        features = tf.layers.batch_normalization(
            features, training=is_training)

    # Add dropout layer if enabled and not first convolution layer.
    if index == 0 and params.dropout:
        features = tf.layers.dropout(features,
                                     rate=params.dropout,
                                     training=is_training)
    return tf.layers.conv1d(
        features,
        filters=params.num_conv[index],
        kernel_size=params.conv_len[index],
        activation=None,
        strides=1,
        padding="same",
        name="conv1d_%d" % index)


def _add_regular_rnn_layers(convolved, lengths, params):
    """Adds RNN layers."""
    cell = tf.nn.rnn_cell.BasicLSTMCell
    cells_fw = [cell(params.num_nodes) for _ in range(params.num_layers)]
    cells_bw = [cell(params.num_nodes) for _ in range(params.num_layers)]

    if params.dropout > 0.0:
        cells_fw = [tf.contrib.rnn.DropoutWrapper(cell) for cell in cells_fw]
        cells_bw = [tf.contrib.rnn.DropoutWrapper(cell) for cell in cells_bw]

    outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
        cells_fw=cells_fw,
        cells_bw=cells_bw,
        inputs=convolved,
        sequence_length=lengths,
        dtype=tf.float32,
        scope="rnn_classification")

    return outputs


def _add_rnn_layers(features, lengths, params):
    """Adds recurrent neural network layers depending on the cell type."""
    outputs = _add_regular_rnn_layers(features, lengths, params)
    # outputs is [batch_size, L, N] where L is the maximal sequence length and
    # N the number of nodes in the last layer.
    mask = tf.tile(
        tf.expand_dims(tf.sequence_mask(lengths, tf.shape(outputs)[1]), 2),
        [1, 1, tf.shape(outputs)[2]])
    zero_outside = tf.where(mask, outputs, tf.zeros_like(outputs))

    return tf.reduce_sum(zero_outside, axis=1)


def rnn_model(features, num_classes=N_CLASSES, params=None, is_training=False):
    """Model function for RNN classifier.

    This function sets up a neural network which applies convolutional layers
    (as configured with params.num_conv and params.conv_len) to the input.
    The output of the convolutional layers is given to LSTM layers (as
    configured with params.num_layers and params.num_nodes).
    The final state of the all LSTM layers are concatenated and fed to a fully
    connected layer to obtain the final classification scores.

    Args:
      features: dictionary with keys: inks, lengths.
      num_classes: number of classes
      params: a parameter dictionary with the following keys: num_layers,
        num_nodes, batch_size, num_conv, conv_len, num_classes.
      is_training: True when training the model

    Returns:
      ModelFnOps for Estimator API.
    """
    if params is None:
        params = DEFAULT_PARAMS

    lengths = _sequence_length(features)
    for i in range(len(params.num_conv)):
        features = _add_conv_layers(features, params, i, is_training)

    final_state = _add_rnn_layers(features, lengths, params)
    logits = tf.layers.dense(final_state, num_classes)

    return final_state
