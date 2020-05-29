import os
import argparse
import numpy as np
import tensorflow as tf
import re
import cv2
import datetime

from model import Model
from utils.Dataloader import MJSynthDataset
from utils.label_converter import AttnLabelConverter, CTCLabelConverter
from utils.accuracy import accuracy_match


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_folder', default="./dataset/mnt", type=str, help='folder path to input images')
    # Training Parameter
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=100)  # batch size for training
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
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512, help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
    args = parser.parse_args()

    # log configure
    log_filename = "test_%s.txt" % datetime.datetime.now().strftime("%Y_%m_%d")
    log_dir = os.path.join(args.log_dir, "test")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    summary_writer = tf.summary.create_file_writer(log_dir)
    tf.summary.trace_on(graph=True, profiler=True)

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

        # https://heartbeat.fritz.ai/building-a-data-pipeline-with-tensorflow-3047656b5095
        root_path = os.path.abspath(os.path.join(args.test_folder, "ramdisk/max/90kDICT32px"))
        with open(os.path.join(root_path, "annotation_test.txt"), "r") as fr:
            raw_data = fr.readlines()
        image_path_list = [os.path.join(root_path, re.match("./(.*.jpg)(.*)", image_path).group(1)) for image_path in raw_data]
        total_data_size = len(image_path_list)
        dataset = MJSynthDataset(image_path_list, (args.imgH, args.imgW, args.input_channel))
        dataset = dataset.batch(args.batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        counter = 1
        gts_list, preds_list, trans_list = list(), list(), list()
        for batch_idx, data in enumerate(dataset):
            image_tensors, labels, paths = data
            labels_list = [label[0] for label in np.asarray(labels, dtype=str)]
            paths = np.asarray(paths)
            batch_size = np.shape(image_tensors)[0]
            decoded_labels = [re.match("(.*)_(.*)_(.*)", str(path)).group(2) for path in paths]

            if "Attn" in args.Prediction:
                length_for_pred = tf.zeros((batch_size, args.batch_max_length), dtype=tf.int32)
            elif "CTC" in args.Prediction:
                length_for_pred = tf.multiply(tf.ones(batch_size, dtype=tf.int32), args.batch_max_length + 1)
            else:
                raise NotImplementedError

            text, length = converter.encode(decoded_labels, batch_max_length=args.batch_max_length)

            if 'CTC' in args.Prediction:
                trans, preds = net(image_tensors, text)

                # ignore [B] token => ignore index 0
                losses = tf.nn.ctc_loss(text, preds, length,
                                        tf.multiply(tf.ones(tf.shape(preds)[0]), args.batch_max_length),
                                        logits_time_major=False, blank_index=0)
                # raise NotImplementedError
            else:
                trans, preds = net(image_tensors, text[:, :-1], is_train=True)  # align with Attention.forward
                target = text[:, 1:]  # without [B] Symbol
                target_onehot = tf.one_hot(target, args.num_class)

                # ignore [B] token => ignore index 0
                mask = (target != 0)
                losses = tf.nn.softmax_cross_entropy_with_logits(target_onehot, preds)
                losses = tf.where(mask, losses, tf.zeros_like(losses))

            # decode output string
            if "Attn" in args.Prediction:
                preds_index = tf.argmax(preds, axis=-1)
                preds_str = converter.decode(preds_index, length_for_pred)
            elif "CTC" in args.Prediction:
                preds_str = converter.decode(preds, length_for_pred)

            loss = tf.nn.compute_average_loss(losses)

            preds_list.extend(preds_str)
            gts_list.extend(labels_list)
            trans_list.extend(trans)

            print("batch_index: (%d, %d), loss: %f, accuracy: %f" % (batch_idx,
                (total_data_size // args.batch_size) + (1 if total_data_size % args.batch_size > 0 else 0),
                loss, accuracy_match(labels_list, preds_str)))
            # for idx in range(batch_size):
            #     # https://docs.python.org/3/library/re.html
            #     filename = re.match("(.*)/(.*)(\..*)", str(paths[idx][0])).group(2)
            #     print("%s, text: %s, length: %d" % (filename, preds_str[idx], len(preds_str[idx].replace("[B]", "B").replace("[E]", "E"))))
            #     cv2.imwrite("out_%d.jpg" % counter, np.asarray(trans[idx], np.int))
            #     counter += 1

        # calculate accuracy
        print("accuracy: %f" % accuracy_match(gts_list, preds_list))

        # for idx in range(len(gts_list)):
        #     print("label: %s, text: %s, length: %d" % (gts_list[idx], preds_list[idx], len(preds_list[idx])))
        #     cv2.imwrite("results/out_%d.jpg" % counter, np.asarray(trans_list[idx], np.int))
        #     counter += 1

        # log
        with summary_writer.as_default():
            tf.summary.trace_export(name="func_trace", step=0, profiler_outdir=log_dir)
        print("ok")
