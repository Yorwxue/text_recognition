import os
import itertools
from collections import defaultdict
from PIL import Image
import tensorflow as tf
import cv2
import numpy as np
from natsort import natsorted
from utils.img_utils import ratioImputation


class RawDataset(tf.data.Dataset):
    # OUTPUT: (steps, timings, counters)
    _INSTANCES_COUNTER = itertools.count()  # Number of datasets generated
    _EPOCHS_COUNTER = defaultdict(itertools.count)  # Number of epochs done for each dataset

    def _generator(self, imgH, imgW, ch, image_path_list, num_samples):

        for sample_idx in range(num_samples):
            print(image_path_list[sample_idx])
            try:
                img = np.array(Image.open(image_path_list[sample_idx]), dtype=np.float32)[:, :, :3]  # rgb
                img = ratioImputation(img, (imgH, imgW))
                # cv2.imwrite("test.jpg", img)
                if ch == 1:
                    img = np.expand_dims(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), axis=-1)

            except IOError:
                print(f'Corrupted image for {sample_idx}')
                # make dummy image and dummy label for corrupted image.
                img = np.zeros((imgW, imgH, ch), dtype=np.float32)

            yield (img, [image_path_list[sample_idx]])

    def __new__(cls, root, shape):
        imgH, imgW, ch = shape
        image_path_list = list()
        for dirpath, dirnames, filenames in os.walk(root):
            for name in filenames:
                _, ext = os.path.splitext(name)
                ext = ext.lower()
                if ext == '.jpg' or ext == '.jpeg' or ext == '.png':
                    image_path_list.append(os.path.join(dirpath, name))

        image_path_list = natsorted(image_path_list)
        nSamples = len(image_path_list)

        return tf.data.Dataset.from_generator(
            cls._generator,
            output_types=(tf.dtypes.float32, tf.dtypes.string),
            output_shapes=((imgH, imgW, ch), 1),
            args=(next(cls._INSTANCES_COUNTER), imgH, imgW, ch, image_path_list, nSamples)
        )


if __name__ == "__main__":
    root = os.path.abspath("../images")
    raw_data = RawDataset(root, (32, 100, 3))
    for idx, data in enumerate(raw_data):
        img, path = data
        np_img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        cv2.imshow("%d" % idx, np_img)
    cv2.waitKey(5000)
    print()
