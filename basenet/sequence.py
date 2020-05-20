import tensorflow as tf


class BidirectionalLSTM(tf.keras.Model):
    def __init__(self, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.forward_layer = tf.keras.layers.LSTM(hidden_size, return_sequences=True)
        self.baxkward_layer = tf.keras.layers.LSTM(hidden_size, return_sequences=True, go_backwards=True)
        self.rnn = tf.keras.layers.Bidirectional(self.forward_layer, backward_layer=self.baxkward_layer, merge_mode="concat")
        self.nn = tf.keras.layers.Dense(output_size)

    def call(self, x):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        recurrent = self.rnn(x)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        output = self.nn(recurrent)  # batch_size x T x output_size
        return output
