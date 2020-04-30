import os
import itertools
from collections import defaultdict
from PIL import Image
import tensorflow as tf
import cv2
import numpy as np
from natsort import natsorted
import re
from utils.img_utils import ratioImputation


class RawDataset(tf.data.Dataset):
    # OUTPUT: (steps, timings, counters)
    _INSTANCES_COUNTER = itertools.count()  # Number of datasets generated
    _EPOCHS_COUNTER = defaultdict(itertools.count)  # Number of epochs done for each dataset

    def _generator(self, imgH, imgW, ch, image_path_list, num_samples):

        for sample_idx in range(num_samples):
            # print(image_path_list[sample_idx])
            try:
                img = np.array(Image.open(image_path_list[sample_idx]), dtype=np.float32)[:, :, :3]  # rgb
                img = ratioImputation(img, (imgH, imgW))
                # cv2.imwrite("test.jpg", img)
                if ch == 1:
                    img = np.expand_dims(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), axis=-1)

            except IOError:
                print(f'Corrupted image for {sample_idx}')
                # make dummy image and dummy label for corrupted image.
                img = np.zeros((imgH, imgW, ch), dtype=np.float32)

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


class MJSynthDataset(tf.data.Dataset):
    # OUTPUT: (steps, timings, counters)
    _INSTANCES_COUNTER = itertools.count()  # Number of datasets generated
    _EPOCHS_COUNTER = defaultdict(itertools.count)  # Number of epochs done for each dataset

    def _generator(self, imgH, imgW, ch, image_path_list, num_samples):
        for sample_idx in range(num_samples):
            image_path = (image_path_list[sample_idx]).decode("utf-8")
            try:
                label = re.match("(.*)_(.*)_(.*)", image_path).group(2)
                img = np.array(Image.open(image_path), dtype=np.float32)[:, :, :3]  # rgb
                img = ratioImputation(img, (imgH, imgW))
                if ch == 1:
                    img = np.expand_dims(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), axis=-1)
            except Exception as e:
                print(e, end="")
                print(f', corrupted image for {sample_idx}')
                # make dummy image and dummy label for corrupted image.
                img = np.zeros((imgH, imgW, ch), dtype=np.float32)
                label = ""

            yield (img, [label], [image_path])

    def __new__(cls, root, shape):
        imgH, imgW, ch = shape
        image_path_list = list()
        count = 0
        for dirpath, dirnames, filenames in os.walk(root, topdown=False):
            for name in filenames:
                _, ext = os.path.splitext(name)
                ext = ext.lower()
                if ext == '.jpg' or ext == '.jpeg' or ext == '.png':
                    image_path_list.append(os.path.join(dirpath, name))
                    count += 1
            if count >= 100:
                break

        image_path_list = natsorted(image_path_list)
        nSamples = len(image_path_list)

        return tf.data.Dataset.from_generator(
            cls._generator,
            output_types=(tf.dtypes.float32, tf.dtypes.string, tf.dtypes.string),
            output_shapes=((imgH, imgW, ch), 1, 1),
            args=(next(cls._INSTANCES_COUNTER), imgH, imgW, ch, image_path_list, nSamples)
        )


if __name__ == "__main__":
    batch_size = 5
    max_length = 25
    # root = os.path.abspath("../images")
    # dataset = RawDataset(root, (32, 100, 3))

    root = os.path.abspath("../dataset/mnt")
    dataset = MJSynthDataset(root, (32, 100, 3))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    for batch_idx, datas in enumerate(dataset):
        imgs, labels, paths = datas
        num_data = np.asarray(tf.shape(datas[0]))[0]
        for idx in range(num_data):
            try:
                label = np.asarray(labels[idx])[0]
                # label = ("".join([char.decode("utf-8") for char in label])).replace("[B]", "").replace("[E]", "")
                label = label[0].decode("utf-8")
                np_img = cv2.cvtColor(np.asarray(imgs[idx]), cv2.COLOR_RGB2BGR)
                cv2.imwrite("../results/%d_%d_%s.jpg" % (batch_idx, idx, label), np_img)
            except Exception as e:
                print(e)
    print()
