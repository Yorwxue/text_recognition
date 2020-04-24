import os
import argparse
import cv2
import numpy as np
import tensorflow as tf

from model import Model
from utils.Dataloader import RawDataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_folder', default='./images/', type=str, help='folder path to input images')
    # Training Parameter
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=2)  # batch size for training
    parser.add_argument('--iterations', '--iter', type=int, default=100000)
    parser.add_argument('--weight_dir', type=str, default=r"./weights/", help="directory to save model weights")
    # Model Architecture
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--F', type=int, default=20, help="number of fiducial points of TPS-STN")
    parser.add_argument('--Transformation', type=str, default="TPS", help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, default="ResNet", help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, default="BiLSTM", help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, default="Attn", help='Prediction stage. CTC|Attn')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512, help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
    args = parser.parse_args()

    net = Model(args)

    filenames = os.listdir(args.test_folder)
    # https://heartbeat.fritz.ai/building-a-data-pipeline-with-tensorflow-3047656b5095
    raw_data = RawDataset(os.path.abspath("./images"), (args.imgH, args.imgW, args.input_channel))
    raw_data = raw_data.batch(args.batch_size)
    raw_data = raw_data.prefetch(tf.data.experimental.AUTOTUNE)

    for data in raw_data:
        img_tensor, path = data
        batch_size = np.shape(img_tensor)[0]

        # # show image
        # img = np.asarray(img)
        # print(np.shape(img))
        # if args.input_channel == 3:
        #     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # cv2.imshow("%d" % idx, img)
        # cv2.waitKey(3000)

        text = tf.zeros((batch_size, args.batch_max_length + 1))
        net(img_tensor, text, is_train=False)
