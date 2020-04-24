import tensorflow as tf


class Attention(tf.keras.Model):
    """
    Attention Decoder that used to decode output of the sequence classifications of Encoder.
    More detail can be found in https://arxiv.org/abs/1603.03915
    """
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention_cell = AttentionCell(hidden_size)

    def call(self, batch_H, text, is_train=True, batch_max_length=25):
        """
                input:
                    batch_H : contextual_feature H = hidden state of encoder. [batch_size x num_steps x num_classes]
                    text : the text-index of each image. [batch_size x (max_length+1)]. +1 for [B] token. text[:, 0] = [B].
                output: probability distribution at each step [batch_size x num_steps x num_classes]
                """
        batch_size = batch_H.shape[0]


class AttentionCell(tf.keras.Model):
    def __init__(self, hidden_size):
        """
        Structure of Attention Decoder can be found in https://arxiv.org/abs/1603.03915, figure.5
        """
        super(AttentionCell, self).__init__()
        self.i2h = tf.keras.layers.Dense(hidden_size, use_bias=False)
        self.h2h = tf.keras.layers.Dense(hidden_size)
        self.score = tf.keras.layers.Dense(1, use_bias=False)
        self.rnn = tf.keras.layers.LSTMCell(hidden_size)
        self.hidden_size = hidden_size

    def call(self, prev_hidden, batch_H, char_onehots):
        """
        [batch_size x num_encoder_step x num_channel] -> [batch_size x num_encoder_step x hidden_size]

        :param prev_hidden: previous hidden state
        :param batch_H: output of sequence model such as LSTM in Encoder
        :param char_onehots:
        :return:
        """
        batch_H_proj = self.i2h(batch_H)
        prev_hidden_proj = self.h2h(prev_hidden[0]).unsqueeze(1)
        e = self.score(tf.keras.activations.tanh(batch_H_proj + prev_hidden_proj))  # batch_size x num_encoder_step * 1
        alpha = tf.keras.activations.softmax(e, dim=1)

        # element-wise multiply of attention score with batch_H
        context = tf.matmul(alpha.permute(0, 2, 1), batch_H).squeeze(1)  # batch_size x num_channel

        concat_context = tf.concat([context, char_onehots], 1)  # batch_size x (num_channel + num_embedding)
        cur_hidden = self.rnn(concat_context, prev_hidden)
        return cur_hidden, alpha
