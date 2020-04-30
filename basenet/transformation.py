import tensorflow as tf
import numpy as np

class TPS_SpatialTransformerNetwork(tf.keras.Model):
    def __init__(self, F, I_r_size):
        """
        Rectification Network of RARE, namely TPS based STN
        :param F: number of fiducial points
        """
        super(TPS_SpatialTransformerNetwork, self).__init__()
        self.F = F
        self.I_r_size = I_r_size  # = (I_r_height, I_r_width)
        self.LocalizationNetwork = LocalizationNet(self.F)
        self.GridGenerator = GridGenerator(self.F, self.I_r_size)

    def call(self, batch_I):
        batch_C_prime = self.LocalizationNetwork(batch_I)
        build_P_prime = self.GridGenerator.build_P_prime(batch_C_prime)
        build_P_prime_reshape = tf.reshape(build_P_prime, ([build_P_prime.shape[0], self.I_r_size[0], self.I_r_size[1], 2]))

        batch_I_r = bilinear_sampler(batch_I, build_P_prime_reshape[:, :, :, 0], build_P_prime_reshape[:, :, :, 1])
        return batch_I_r


class LocalizationNet(tf.keras.Model):
    def __init__(self, F):
        """
        Localization Network of RARE, which predicts C' (K x 2) from I (I_width x I_height)
        :param F: number of fiducial points
        """
        super(LocalizationNet, self).__init__()
        self.F = F

        # fc2 initializer
        init_kernel = tf.random_uniform_initializer(0, 0)
        ctrl_pts_x = np.linspace(-1.0, 1.0, int(self.F / 2))
        ctrl_pts_y_top = np.linspace(0.0, -1.0, num=int(self.F / 2))
        ctrl_pts_y_bottom = np.linspace(1.0, 0.0, num=int(self.F / 2))
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        init_bias = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        init_bias = tf.constant_initializer(init_bias)

        self.bn = tf.keras.layers.BatchNormalization()
        self.Pool = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))
        self.Conv_1 = tf.keras.layers.Conv2D(64, (3, 3), use_bias=False, padding="same", activation="relu")
        self.Conv_2 = tf.keras.layers.Conv2D(128, (3, 3), use_bias=False, padding="same", activation="relu")
        self.Conv_3 = tf.keras.layers.Conv2D(256, (3, 3), use_bias=False, padding="same", activation="relu")
        self.Conv_4 = tf.keras.layers.Conv2D(512, (3, 3), use_bias=False, padding="same", activation="relu")
        self.FC_1 = tf.keras.layers.Dense(256)
        self.FC_2 = tf.keras.layers.Dense(self.F * 2, kernel_initializer=init_kernel, bias_initializer=init_bias)

    def call(self, x):
        h = self.Conv_1(x)
        # h = self.bn(h)
        h = self.Pool(h)
        h = self.Conv_2(h)
        # h = self.bn(h)
        h = self.Pool(h)
        h = self.Conv_3(h)
        # h = self.bn(h)
        h = self.Pool(h)
        h = self.Conv_4(h)
        # h = self.bn(h)
        h = tf.keras.layers.AveragePooling2D(pool_size=(h.shape[1], h.shape[2]))(h)
        h = tf.squeeze(h, axis=[1, 2])
        h = self.FC_1(h)
        h = self.FC_2(h)
        y = tf.reshape(h, (tf.shape(h)[0], self.F, 2))
        return y


class GridGenerator(tf.keras.Model):
    """
    Grid Generator of RARE, which produces P_prime by multipling T with P
    https://github.com/clovaai/deep-text-recognition-benchmark
    """

    def __init__(self, F, I_r_size):
        """ Generate P_hat and inv_delta_C for later """
        super(GridGenerator, self).__init__()
        self.eps = 1e-6
        self.I_r_height, self.I_r_width = I_r_size
        self.F = F
        self.C = self._build_C(self.F)  # F x 2
        self.P = self._build_P(self.I_r_width, self.I_r_height)

        self.inv_delta_C = tf.convert_to_tensor(self._build_inv_delta_C(self.F, self.C), dtype=tf.float32)  # F+3 x F+3
        self.P_hat = tf.convert_to_tensor(self._build_P_hat(self.F, self.C, self.P), dtype=tf.float32)  # n x F+3

    def _build_C(self, F):
        """ Return coordinates of fiducial points in I_r; C """
        ctrl_pts_x = np.linspace(-1.0, 1.0, int(F / 2))
        ctrl_pts_y_top = -1 * np.ones(int(F / 2))
        ctrl_pts_y_bottom = np.ones(int(F / 2))
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        C = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        return C  # F x 2

    def _build_inv_delta_C(self, F, C):
        """ Return inv_delta_C which is needed to calculate T """
        hat_C = np.zeros((F, F), dtype=float)  # F x F
        for i in range(0, F):
            for j in range(i, F):
                r = np.linalg.norm(C[i] - C[j])
                hat_C[i, j] = r
                hat_C[j, i] = r
        np.fill_diagonal(hat_C, 1)
        hat_C = (hat_C ** 2) * np.log(hat_C)
        # print(C.shape, hat_C.shape)
        delta_C = np.concatenate(  # F+3 x F+3
            [
                np.concatenate([np.ones((F, 1)), C, hat_C], axis=1),  # F x F+3
                np.concatenate([np.zeros((2, 3)), np.transpose(C)], axis=1),  # 2 x F+3
                np.concatenate([np.zeros((1, 3)), np.ones((1, F))], axis=1)  # 1 x F+3
            ],
            axis=0
        )
        inv_delta_C = np.linalg.inv(delta_C)
        return inv_delta_C  # F+3 x F+3

    def _build_P(self, I_r_width, I_r_height):
        I_r_grid_x = (np.arange(-I_r_width, I_r_width, 2) + 1.0) / I_r_width  # self.I_r_width
        I_r_grid_y = (np.arange(-I_r_height, I_r_height, 2) + 1.0) / I_r_height  # self.I_r_height
        P = np.stack(  # self.I_r_width x self.I_r_height x 2
            np.meshgrid(I_r_grid_x, I_r_grid_y),
            axis=2
        )
        return P.reshape([-1, 2])  # n (= self.I_r_width x self.I_r_height) x 2

    def _build_P_hat(self, F, C, P):
        n = P.shape[0]  # n (= self.I_r_width x self.I_r_height)
        P_tile = np.tile(np.expand_dims(P, axis=1), (1, F, 1))  # n x 2 -> n x 1 x 2 -> n x F x 2
        C_tile = np.expand_dims(C, axis=0)  # 1 x F x 2
        P_diff = P_tile - C_tile  # n x F x 2
        rbf_norm = np.linalg.norm(P_diff, ord=2, axis=2, keepdims=False)  # n x F
        rbf = np.multiply(np.square(rbf_norm), np.log(rbf_norm + self.eps))  # n x F
        P_hat = np.concatenate([np.ones((n, 1)), P, rbf], axis=1)
        return P_hat  # n x F+3

    def build_P_prime(self, batch_C_prime):
        """ Generate Grid from batch_C_prime [batch_size x F x 2] """
        batch_size = batch_C_prime.shape[0]
        # batch_inv_delta_C = self.inv_delta_C.repeat(batch_size, 1, 1)
        batch_inv_delta_C = tf.tile([self.inv_delta_C], (batch_size, 1, 1))
        # batch_P_hat = self.P_hat.repeat(batch_size, 1, 1)
        batch_P_hat = tf.tile([self.P_hat], (batch_size, 1, 1))
        batch_C_prime_with_zeros = tf.concat((batch_C_prime, tf.zeros((batch_size, 3, 2), dtype=np.float32)), axis=1)  # batch_size x F+3 x 2
        batch_T = tf.matmul(batch_inv_delta_C, batch_C_prime_with_zeros)  # batch_size x F+3 x 2
        batch_P_prime = tf.matmul(batch_P_hat, batch_T)  # batch_size x n x 2
        return batch_P_prime  # batch_size x n x 2


def bilinear_sampler(img, x, y):
    """
    Performs bilinear sampling of the input images according to the
    normalized coordinates provided by the sampling grid. Note that
    the sampling is done identically for each channel of the input.
    To test if the function works properly, output image should be
    identical to input image when theta is initialized to identity
    transform.
    ref.: https://github.com/kevinzakka/spatial-transformer-network/blob/master/stn/transformer.py#L159

    Input
    -----
    - img: batch of images in (B, H, W, C) layout.
    - grid: x, y which is the output of affine_grid_generator.
    Returns
    -------
    - out: interpolated images according to grids. Same size as grid.
    """
    H = tf.shape(img)[1]
    W = tf.shape(img)[2]
    max_y = tf.cast(H - 1, 'int32')
    max_x = tf.cast(W - 1, 'int32')
    zero = tf.zeros([], dtype='int32')

    # rescale x and y to [0, W-1/H-1]
    x = tf.cast(x, 'float32')
    y = tf.cast(y, 'float32')
    x = 0.5 * ((x + 1.0) * tf.cast(max_x-1, 'float32'))
    y = 0.5 * ((y + 1.0) * tf.cast(max_y-1, 'float32'))

    # grab 4 nearest corner points for each (x_i, y_i)
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1

    # clip to range [0, H-1/W-1] to not violate img boundaries
    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)

    # get pixel value at corner coords
    Ia = get_pixel_value(img, x0, y0)
    Ib = get_pixel_value(img, x0, y1)
    Ic = get_pixel_value(img, x1, y0)
    Id = get_pixel_value(img, x1, y1)

    # recast as float for delta calculation
    x0 = tf.cast(x0, 'float32')
    x1 = tf.cast(x1, 'float32')
    y0 = tf.cast(y0, 'float32')
    y1 = tf.cast(y1, 'float32')

    # calculate deltas
    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    # add dimension for addition
    wa = tf.expand_dims(wa, axis=3)
    wb = tf.expand_dims(wb, axis=3)
    wc = tf.expand_dims(wc, axis=3)
    wd = tf.expand_dims(wd, axis=3)

    # compute output
    out = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])

    return out


def get_pixel_value(img, x, y):
    """
    Utility function to get pixel value for coordinate
    vectors x and y from a  4D tensor image.
    Input
    -----
    - img: tensor of shape (B, H, W, C)
    - x: flattened tensor of shape (B*H*W,)
    - y: flattened tensor of shape (B*H*W,)
    Returns
    -------
    - output: tensor of shape (B, H, W, C)
    """
    shape = tf.shape(x)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]

    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
    b = tf.tile(batch_idx, (1, height, width))

    indices = tf.stack([b, y, x], 3)

    return tf.gather_nd(img, indices)


if __name__ == "__main__":
    import numpy as np
    x = np.random.uniform(0, 255, size=(1, 100, 32, 1))
    net = LocalizationNet(5)
    y = net(x)
    print()
