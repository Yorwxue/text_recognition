import os
import argparse
import numpy as np
import tensorflow as tf
import re
import datetime
import cv2

from model import Model
from utils.Dataloader import MJSynthDataset
from utils.label_converter import AttnLabelConverter, CTCLabelConverter
from utils.accuracy import accuracy_match


if __name__ == "__main__":
    # gpu setting
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    if gpu_devices:
        tf.config.experimental.set_memory_growth(gpu_devices[0], True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_folder', default='./images/', type=str, help='folder path to input images')
    # Training Parameter
    parser.add_argument('--pretrained', type=bool, default=True, help='fine-tune from pre-trained model')              # start from pre-trained model
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=100)  # batch size for training
    parser.add_argument('--iterations', '--iter', type=int, default=100000)
    parser.add_argument('--weight_dir', type=str, default=r"./weights/", help="directory to save model weights")
    parser.add_argument('--log_dir', type=str, default=r"./logs/", help="directory to save logs")
    parser.add_argument('--valid_size', type=int, default=100, help="size of validation dataset")

    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate, default=1.0 for Adadelta')               # 1e-3 for CTC, 1 for Attn

    parser.add_argument('--rho', type=float, default=0.95, help='decay rate rho for Adadelta. default=0.95')
    parser.add_argument('--eps', type=float, default=1e-8, help='eps for Adadelta. default=1e-8')
    parser.add_argument('--epochs', type=int, default=300000, help='number of epochs')
    # Model Architecture
    parser.add_argument('--character', type=str, default='./config/character', help='path to definition of character label')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--F', type=int, default=20, help="number of fiducial points of TPS-STN")
    parser.add_argument('--Transformation', type=str, default="TPS", help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, default="ResNet", help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, default="BiLSTM", help='SequenceModeling stage. None|BiLSTM')

    parser.add_argument('--Prediction', type=str, default="CTC", help='Prediction stage. CTC|Attn')                     # CTC or Attn

    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512, help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
    args = parser.parse_args()

    # log configure
    log_filename = "train_%s.txt" % datetime.datetime.now().strftime("%Y_%m_%d")
    log_dir = os.path.join(args.log_dir, "train")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    summary_writer = tf.summary.create_file_writer(log_dir)

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

    # optimizer
    optimizer = tf.keras.optimizers.Adadelta(learning_rate=args.lr, rho=args.rho, epsilon=args.eps)

    # saving configure
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=net)
    manager = tf.train.CheckpointManager(
        checkpoint, directory=args.weight_dir, max_to_keep=10)
    # Create a checkpoint directory to store the checkpoints.
    if not os.path.exists(args.weight_dir):
        os.makedirs(args.weight_dir)
    # checkpoint_dir = os.path.join(args.weight_dir, "ckpt")
    # checkpoint_prefix = os.path.abspath(checkpoint_dir)

    # model restore
    if args.pretrained:
        checkpoint_dir = tf.train.latest_checkpoint(args.weight_dir)
        # checkpoint_dir = os.path.join(args.weight_dir, "ckpt-10")
        checkpoint.restore(checkpoint_dir)
        print("Restored from %s" % checkpoint_dir)

    # dataset
    print("Preparing training data ..")
    filenames = os.listdir(args.test_folder)
    # https://heartbeat.fritz.ai/building-a-data-pipeline-with-tensorflow-3047656b5095
    root_path = os.path.abspath(os.path.join("./dataset/mnt", "ramdisk/max/90kDICT32px"))
    with open(os.path.join(root_path, "annotation_train.txt"), "r") as fr:
        raw_data = fr.readlines()
    image_path_list = [os.path.join(root_path, re.match("./(.*.jpg)(.*)", image_path).group(1)) for image_path in raw_data]
    total_data_size = len(image_path_list)
    dataset = MJSynthDataset(image_path_list, (args.imgH, args.imgW, args.input_channel))
    dataset = dataset.batch(args.batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # valid dataset
    print("Preparing validation data ..")
    with open(os.path.join(root_path, "annotation_val.txt"), "r") as fr:
        raw_valid_data = fr.readlines()
    valid_image_path_list = [os.path.join(root_path, re.match("./(.*.jpg)(.*)", image_path).group(1)) for image_path in raw_valid_data]
    valid_dataset = MJSynthDataset(valid_image_path_list, (args.imgH, args.imgW, args.input_channel), limitation=args.valid_size)
    valid_dataset = valid_dataset.batch(args.batch_size)
    valid_dataset = valid_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    counter = 0
    print("Start training ..")
    for epoch_idx in range(args.epochs):
        for batch_idx, data in enumerate(dataset):
            with tf.GradientTape() as tape:
                image_tensors, labels, paths = data
                decoded_labels = [[char.decode("utf-8") for char in np.asarray(label)] for label in labels]
                text, length = converter.encode(decoded_labels, batch_max_length=args.batch_max_length)

                if 'CTC' in args.Prediction:
                    trans, preds = net(image_tensors, text)

                    # ignore [B] token => ignore index 0
                    losses = tf.nn.ctc_loss(text, preds, length, tf.multiply(tf.ones(tf.shape(preds)[0]), args.batch_max_length), logits_time_major=False, blank_index=0)
                    # raise NotImplementedError
                else:
                    trans, preds = net(image_tensors, text[:, :-1], is_train=True)  # align with Attention.forward
                    target = text[:, 1:]  # without [B] Symbol
                    target_onehot = tf.one_hot(target, args.num_class)

                    # ignore [B] token => ignore index 0
                    mask = (target != 0)
                    losses = tf.nn.softmax_cross_entropy_with_logits(target_onehot, preds)
                    losses = tf.where(mask, losses, tf.zeros_like(losses))

                    # save transformed image
                    # cv2.imwrite("in.jpg", np.asarray(image_tensors[0], np.int))
                    # cv2.imwrite("out.jpg", np.asarray(trans[0], np.int))

                    # show result
                    # target_idx = tf.argmax(target_onehot, axis=-1)
                    # preds_index = tf.argmax(preds, axis=-1)
                    # batch_size = tf.shape(preds)[0]
                    # length_for_pred = tf.zeros((batch_size, args.batch_max_length))
                    # preds_str = converter.decode(preds_index, length_for_pred)
                    # target_str = converter.decode(target_idx, length_for_pred)
                    # for idx in range(batch_size):
                    #     filename = re.match("(.*)/(.*)(\..*)", str(paths[idx][0])).group(2)
                    #     print("%s, target: %s, pred: %s, length: %d" % (filename, target_str[idx], preds_str[idx], len(preds_str[idx].replace("[B]", "B").replace("[E]", "E"))))

            loss = tf.nn.compute_average_loss(losses)

            # training strategy
            #########################################
            # 1. Freeze parameters of Transformation net
            if loss > 1:
                gradients = tape.gradient(losses, net.trainable_variables[8:])
                optimizer.apply_gradients(zip(gradients, net.trainable_variables[8:]))  # without Transformation

            # 2. Apply gradients to all net
            else:
                gradients = tape.gradient(losses, net.trainable_variables)
                optimizer.apply_gradients(zip(gradients, net.trainable_variables))
            #########################################

            # validation
            # """###################################################################
            valid_gts_list, valid_preds_list, valid_trans_list = list(), list(), list()
            for valid_data in valid_dataset:
                valid_image_tensors, valid_labels, valid_paths = valid_data
                valid_decoded_labels = [[char.decode("utf-8") for char in np.asarray(label)] for label in valid_labels]
                valid_text, valid_length = converter.encode(valid_decoded_labels, batch_max_length=args.batch_max_length)
                valid_paths = np.asarray(valid_paths)
                valid_batch_size = np.shape(valid_image_tensors)[0]
                valid_labels = [re.match("(.*)_(.*)_(.*)", str(valid_path)).group(2) for valid_path in valid_paths]

                if "Attn" in args.Prediction:
                    valid_length_for_pred = tf.zeros((valid_batch_size, args.batch_max_length), dtype=tf.int32)
                elif "CTC" in args.Prediction:
                    valid_length_for_pred = tf.multiply(tf.ones(valid_batch_size, dtype=tf.int32), args.batch_max_length + 1)

                # prediction
                ###############
                if 'CTC' in args.Prediction:
                    valid_trans, valid_preds = net(valid_image_tensors, valid_text)

                    # ignore [B] token => ignore index 0
                    valid_losses = tf.nn.ctc_loss(valid_text, valid_preds, valid_length,
                                            tf.multiply(tf.ones(tf.shape(valid_preds)[0]), args.batch_max_length),
                                            logits_time_major=False, blank_index=0)
                else:
                    valid_trans, valid_preds = net(valid_image_tensors, valid_text[:, :-1], is_train=True)  # align with Attention.forward
                    valid_target = valid_text[:, 1:]  # without [B] Symbol
                    valid_target_onehot = tf.one_hot(valid_target, args.num_class)

                    # ignore [B] token => ignore index 0
                    valid_mask = (target != 0)
                    valid_losses = tf.nn.softmax_cross_entropy_with_logits(valid_target_onehot, valid_preds)
                    valid_losses = tf.where(valid_mask, valid_losses, tf.zeros_like(valid_losses))
                ###############

                if "Attn" in args.Prediction:
                    valid_preds_index = tf.argmax(valid_preds, axis=-1)
                    valid_preds_str = converter.decode(valid_preds_index, valid_length_for_pred)
                elif "CTC" in args.Prediction:
                    valid_preds_str = converter.decode(valid_preds, valid_length_for_pred)
                valid_preds_list.extend(valid_preds_str)
                valid_gts_list.extend(valid_labels)
                valid_trans_list.extend(valid_trans)

            accuracy = accuracy_match(valid_gts_list, valid_preds_str)
            valid_loss = tf.nn.compute_average_loss(valid_losses)
            ####################################################################"""

            print("epoch_index: %d, batch_index: (%d, %d), train_loss: %f, valid_loss: %f" % (
                epoch_idx, batch_idx,
                (total_data_size//args.batch_size) + (1 if total_data_size%args.batch_size > 0 else 0),
                loss, valid_loss))

            if batch_idx % 50 == 0:
                # checkpoint.save(checkpoint_prefix)
                manager.save()

                # logs
                with summary_writer.as_default():
                    tf.summary.scalar('loss', loss, step=counter)
                    # tf.summary.trace_on(graph=True, profiler=True)
                    # tf.summary.trace_export(name="func_trace", step=0, profiler_outdir=log_dir)

                batch_size = np.shape(data[0])[0]

                if "Attn" in args.Prediction:
                    preds_index = tf.argmax(preds, axis=-1)
                    length_for_pred = tf.zeros((batch_size, args.batch_max_length), dtype=tf.int32)
                elif "CTC" in args.Prediction:
                    preds_index = preds
                    length_for_pred = tf.multiply(tf.ones(batch_size, dtype=tf.int32), args.batch_max_length+1)
                else:
                    raise NotImplementedError
                preds_str = converter.decode(preds_index, length_for_pred)
                log = "epoch_%d, batch_(%d, %d)" % (epoch_idx, batch_idx, (total_data_size//args.batch_size) + (1 if total_data_size%args.batch_size > 0 else 0))
                for idx in range(batch_size):
                    # https://docs.python.org/3/library/re.html
                    filename = re.match("(.*)/(.*)(\..*)", str(paths[idx][0])).group(2)
                    log += "%s, gt: %s, pred: %s, " % (
                        filename,
                        decoded_labels[idx][0],
                        preds_str[idx].replace("[B]", "").replace("[E]", "")
                    )

                with open(os.path.join(log_dir, log_filename), "a+") as fw:
                    fw.write(log+'\n')

            counter += 1

        manager.save()
