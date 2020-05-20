import tensorflow as tf


class conv_block(tf.keras.Model):
    def __init__(self, mid_ch, out_ch, times):
        """

        :param mid_ch: hidden layer size
        :param out_ch: output layer size
        :param times: number of block
        """
        super(conv_block, self).__init__()
        self.Conv_1 = tf.keras.layers.Conv2D(mid_ch, (3, 3), padding="same")
        self.Conv_2 = tf.keras.layers.Conv2D(out_ch, (3, 3), padding="same")

        self.net = tf.keras.models.Sequential([])
        for i in range(times):
            self.net.add(self.Conv_1)
            self.net.add(self.Conv_2)

    def call(self, x):
        y = self.net(x)
        return y


class ResNet(tf.keras.Model):
    def __init__(self, output_channel):
        super(ResNet, self).__init__()
        self.output_channel_block = [int(output_channel / 16), int(output_channel / 8), int(output_channel / 4), int(output_channel / 2), output_channel]  # 32, 64, 128, 256, 512
        self.Conv_1 = tf.keras.layers.Conv2D(self.output_channel_block[0], (3, 3), padding="same")
        self.Conv_2 = tf.keras.layers.Conv2D(self.output_channel_block[1], (3, 3), padding="same")
        self.Pool_1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))
        self.Block_1 = conv_block(self.output_channel_block[2], self.output_channel_block[2], 1)
        self.Conv_3 = tf.keras.layers.Conv2D(self.output_channel_block[2], (3, 3), padding="same")
        self.Pool_2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))
        self.Block_2 = conv_block(self.output_channel_block[3], self.output_channel_block[3], 2)
        self.Conv_4 = tf.keras.layers.Conv2D(self.output_channel_block[3], (3, 3), padding="same")
        self.Pool_3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 1))
        self.Block_3 = conv_block(self.output_channel_block[4], self.output_channel_block[3], 5)
        self.Conv_5 = tf.keras.layers.Conv2D(self.output_channel_block[4], (3, 3))
        self.Block_4 = conv_block(self.output_channel_block[4], self.output_channel_block[4], 3)
        self.Conv_6 = tf.keras.layers.Conv2D(filters=self.output_channel_block[4], kernel_size=(2, 2), strides=(2, 1), padding="same")
        self.Conv_7 = tf.keras.layers.Conv2D(filters=self.output_channel_block[4], kernel_size=(2, 2), strides=(1, 1), padding="same")

    def call(self, x):
        h = self.Conv_1(x)
        h = self.Conv_2(h)
        h = self.Pool_1(h)
        h = self.Block_1(h)
        h = self.Conv_3(h)
        h = self.Pool_2(h)
        h = self.Block_2(h)
        h = self.Conv_4(h)
        h = self.Pool_3(h)
        h = tf.keras.layers.ZeroPadding2D(padding=(0, 1))(h)
        h = self.Block_3(h)
        h = self.Conv_5(h)
        h = self.Block_4(h)
        h = self.Conv_6(h)
        h = tf.keras.layers.ZeroPadding2D(padding=(0, 1))(h)
        y = self.Conv_7(h)
        return y


if __name__ == "__main__":
    net = ResNet()
    print()
