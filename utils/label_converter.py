import tensorflow as tf
import numpy as np


class AttnLabelConverter(tf.keras.Model):
    def __init__(self, character):
        """

        :param character: set of the possible characters.
        # [B] for the start token of the attention decoder. [E] for end-of-sentence token.
        """
        super(AttnLabelConverter, self).__init__()
        list_token = ["[B]", "[E]"]
        list_character = list(character)
        self.character = list_token + list_character

        self.dict = {}
        for i, char in enumerate(self.character):
            # print(i, char)
            self.dict[char] = i

    def encode(self, texts, batch_max_length=25):
        """ convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default
        output:
            text : the input of attention decoder. [batch_size x (max_length+2)] +1 for [B] token and +1 for [s] token.
                text[:, 0] is [B] token and text is padded with [B] token after [E] token.
            length : the length of output of attention decoder, which count [E] token also. [3, 7, ....] [batch_size]
        """
        lengths = [len(s[0]) + 1 for s in texts]  # +1 for [E] at end of sentence.
        # batch_max_length = max(lengths) # this is not allowed for multi-gpu setting
        batch_max_length += 1
        # additional +1 for [B] at first step. batch_text is padded with [B] token after [E] token.
        batch_text = tf.zeros((len(texts), batch_max_length + 1), dtype=tf.int32)

        for i, t in enumerate(texts):
            text = list(t[0])
            text.append('[E]')
            text = [(self.dict[char.lower()] if '[' not in char else self.dict[char]) for char in text]
            # batch_text[i][1:1 + len(text)] = text  # batch_text[:, 0] = [B] token
            indices = tf.stack([[i]*len(text), tf.range(1, 1 + len(text))], axis=1)
            batch_text = tf.tensor_scatter_nd_update(batch_text, indices, text)
        return batch_text, lengths

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            texts.append(text)
        return texts
