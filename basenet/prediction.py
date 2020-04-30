import tensorflow as tf


class Attention(tf.keras.Model):
    """
    Attention Decoder that used to decode output of the sequence classifications of Encoder.
    More detail can be found in https://arxiv.org/abs/1603.03915
    """
    def __init__(self, hidden_size, num_classes):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.attention_cell = AttentionCell(hidden_size)
        self.generator = tf.keras.layers.Dense(num_classes)

    def _char_to_onehot(self, input_char, onehot_dim=38):
        batch_size = tf.shape(input_char)[0]
        indices = tf.stack([
            tf.range(batch_size),
            tf.cast(input_char, dtype=tf.int32)
        ], axis=1)
        one_hot = tf.zeros((batch_size, onehot_dim), dtype=tf.float32)
        # https://stackoverflow.com/questions/55652981/tensorflow-2-0-how-to-update-tensors
        one_hot = tf.tensor_scatter_nd_update(one_hot, indices, tf.ones(batch_size, dtype=tf.float32))

        return one_hot

    def call(self, batch_H, text, is_train=True, batch_max_length=25):
        """
        input:
            batch_H : contextual_feature H = hidden state of encoder. [batch_size x num_steps x num_classes]
            text : the text-index of each image. [batch_size x (max_length+1)]. +1 for [B] token. text[:, 0] = [B].
        output: probability distribution at each step [batch_size x num_steps x num_classes]
        """
        batch_size = batch_H.shape[0]
        num_steps = batch_max_length + 1  # +1 for [s] at end of sentence.
        output_hiddens = tf.zeros((batch_size, num_steps, self.hidden_size))
        hidden = (tf.zeros((batch_size, self.hidden_size)),
                  tf.zeros((batch_size, self.hidden_size)))

        if is_train:
            for i in range(num_steps):
                # one-hot vectors for a i-th char. in a batch
                char_onehots = self._char_to_onehot(text[:, i], onehot_dim=self.num_classes)
                # hidden : decoder's hidden s_{t-1}, batch_H : encoder's hidden H, char_onehots : one-hot(y_{t-1})
                hidden, alpha = self.attention_cell(hidden, batch_H, char_onehots)

                # output_hiddens[:, i, :] = hidden[0]  # LSTM hidden index (0: hidden, 1: Cell)
                # https://stackoverflow.com/questions/55652981/tensorflow-2-0-how-to-update-tensors
                indices = tf.stack([tf.range(batch_size), tf.tile(tf.constant([i]), [batch_size])], axis=1)
                output_hiddens = tf.tensor_scatter_nd_update(output_hiddens, indices, hidden[0])
            probs = self.generator(output_hiddens)
        else:
            targets = tf.zeros((batch_size))  # [B] token
            probs = tf.zeros((batch_size, num_steps, self.num_classes))

            for i in range(num_steps):
                char_onehots = self._char_to_onehot(targets, onehot_dim=self.num_classes)
                hidden, alpha = self.attention_cell(hidden, batch_H, char_onehots)
                probs_step = self.generator(hidden[0])

                # probs[:, i, :] = probs_step
                # https://stackoverflow.com/questions/55652981/tensorflow-2-0-how-to-update-tensors
                indices = tf.stack([tf.range(batch_size), tf.tile(tf.constant([i]), [batch_size])], axis=1)
                probs = tf.tensor_scatter_nd_update(probs, indices, probs_step)

                next_input = tf.argmax(probs_step, axis=1)
                targets = next_input
        return probs  # batch_size x num_steps x num_classes


class AttentionCell(tf.keras.Model):
    def __init__(self, hidden_size):
        """
        Structure of Attention Decoder can be found in https://arxiv.org/abs/1603.03915, figure.5
        """
        super(AttentionCell, self).__init__()
        self.i2h = tf.keras.layers.Dense(hidden_size, use_bias=False)
        self.h2h = tf.keras.layers.Dense(hidden_size)
        self.score = tf.keras.layers.Dense(1, use_bias=False)
        self.rnn = tf.keras.layers.LSTMCell(hidden_size)  # output of LSTMCell is h, [h, c]
        self.hidden_size = hidden_size

    def call(self, prev_hidden, batch_H, char_onehots):
        """
        [batch_size x num_encoder_step x num_channel] -> [batch_size x num_encoder_step x hidden_size]

        :param prev_hidden: previous hidden state
        :param batch_H: output of sequence model such as LSTM in Encoder
        :param char_onehots: [h, c]
        :return:
        """
        batch_H_proj = self.i2h(batch_H)
        prev_hidden_proj = tf.expand_dims(self.h2h(prev_hidden[0]), axis=1)
        e = self.score(tf.keras.activations.tanh(batch_H_proj + prev_hidden_proj))  # batch_size x num_encoder_step * 1
        alpha = tf.keras.activations.softmax(e, axis=1)

        # element-wise multiply of attention score with batch_H
        context = tf.squeeze(tf.matmul(tf.transpose(alpha, (0, 2, 1)), batch_H), 1)  # batch_size x num_channel

        concat_context = tf.concat([context, char_onehots], 1)  # batch_size x (num_channel + num_embedding)
        cur_hidden = self.rnn(concat_context, prev_hidden)[1]
        return cur_hidden, alpha
