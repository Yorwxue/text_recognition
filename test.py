import os
import argparse
import numpy as np
import tensorflow as tf
import re
import cv2

from model import Model
from utils.Dataloader import RawDataset
from utils.label_converter import AttnLabelConverter, CTCLabelConverter


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_folder', default='./images/', type=str, help='folder path to input images')
    # Training Parameter
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=2)  # batch size for training
    parser.add_argument('--iterations', '--iter', type=int, default=100000)
    parser.add_argument('--weight_dir', type=str, default=r"./weights/", help="directory to save model weights")
    parser.add_argument('--log_dir', type=str, default=r"./logs/", help="directory to save logs")
    # Model Architecture
    parser.add_argument('--character', type=str, default='./config/character', help='path to definition of character label')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--F', type=int, default=20, help="number of fiducial points of TPS-STN")
    parser.add_argument('--Transformation', type=str, default="TPS", help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, default="ResNet", help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, default="BiLSTM", help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, default="CTC", help='Prediction stage. CTC|Attn')
    parser.add_argument('--input_channel', type=int, default=3, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512, help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
    args = parser.parse_args()

    with tf.device('/cpu:0'):
        # model configuration
        with open(args.character, "r") as fr:
            character = fr.readline()
            character = character.replace("\n", "")
        if 'CTC' in args.Prediction:
            converter = CTCLabelConverter(character)
            # raise NotImplementedError
        else:
            converter = AttnLabelConverter(character)

        args.num_class = len(converter.character)
        net = Model(args)

        # model restore
        checkpoint = tf.train.Checkpoint(model=net)
        checkpoint_dir = tf.train.latest_checkpoint(args.weight_dir)
        # checkpoint_dir = os.path.join(args.weight_dir, "ckpt-10")
        checkpoint.restore(checkpoint_dir)
        print("Restored from %s" % checkpoint_dir)

        filenames = os.listdir(args.test_folder)
        # https://heartbeat.fritz.ai/building-a-data-pipeline-with-tensorflow-3047656b5095
        raw_data = RawDataset(os.path.abspath("./images"), (args.imgH, args.imgW, args.input_channel))
        raw_data = raw_data.batch(args.batch_size)
        raw_data = raw_data.prefetch(tf.data.experimental.AUTOTUNE)

        counter = 1
        for data in raw_data:
            img_tensors, paths = data
            paths = np.asarray(paths)
            batch_size = np.shape(img_tensors)[0]

            if "Attn" in args.Prediction:
                length_for_pred = tf.zeros((batch_size, args.batch_max_length), dtype=tf.int32)
            elif "CTC" in args.Prediction:
                length_for_pred = tf.multiply(tf.ones(batch_size, dtype=tf.int32), args.batch_max_length + 1)
            else:
                raise NotImplementedError

            # # show image
            # img = np.asarray(img)
            # print(np.shape(img))
            # if args.input_channel == 3:
            #     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # cv2.imshow("%d" % idx, img)
            # cv2.waitKey(3000)

            text = tf.zeros((batch_size, args.batch_max_length + 1))
            trans, preds = net(img_tensors, text, is_train=False)

            preds_index = tf.argmax(preds, axis=-1)
            preds_str = converter.decode(preds_index, length_for_pred)
            for idx in range(batch_size):
                # https://docs.python.org/3/library/re.html
                filename = re.match("(.*)/(.*)(\..*)", str(paths[idx][0])).group(2)
                print("%s, text: %s, length: %d" % (filename, preds_str[idx], len(preds_str[idx].replace("[B]", "B").replace("[E]", "E"))))
                cv2.imwrite("out_%d.jpg" % counter, np.asarray(trans[idx], np.int))
                counter += 1
