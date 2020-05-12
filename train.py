import os
import argparse
import numpy as np
import tensorflow as tf
import re
import datetime

from model import Model
from utils.Dataloader import MJSynthDataset
from utils.label_converter import AttnLabelConverter


if __name__ == "__main__":
    # gpu setting
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    if gpu_devices:
        tf.config.experimental.set_memory_growth(gpu_devices[0], True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_folder', default='./images/', type=str, help='folder path to input images')
    # Training Parameter
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=128)  # batch size for training
    parser.add_argument('--iterations', '--iter', type=int, default=100000)
    parser.add_argument('--weight_dir', type=str, default=r"./weights/", help="directory to save model weights")
    parser.add_argument('--log_dir', type=str, default=r"./logs/", help="directory to save logs")
    parser.add_argument('--lr', type=float, default=1, help='learning rate, default=1.0 for Adadelta')
    parser.add_argument('--rho', type=float, default=0.95, help='decay rate rho for Adadelta. default=0.95')
    parser.add_argument('--eps', type=float, default=1e-8, help='eps for Adadelta. default=1e-8')
    parser.add_argument('--epochs', type=int, default=100000, help='number of epochs')
    # Model Architecture
    parser.add_argument('--character', type=str, default='./config/character', help='path to definition of character label')
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

    # log configure
    log_filename = "%s.txt" % datetime.datetime.now().strftime("%Y_%m_%d")
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    summary_writer = tf.summary.create_file_writer(args.log_dir)

    # model configuration
    with open(args.character, "r") as fr:
        character = fr.readline()
        character = character.replace("\n", "")
    if 'CTC' in args.Prediction:
        raise NotImplementedError
    else:
        converter = AttnLabelConverter(character)
    args.num_class = len(converter.character)

    net = Model(args)

    # loss
    if 'CTC' in args.Prediction:
        loss_fn = tf.nn.ctc_loss
    else:
        loss_fn = tf.nn.softmax_cross_entropy_with_logits  # TODO: ignore [B] token = ignore index 0

    # filtered_parameters
    # TODO

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

    # dataset
    filenames = os.listdir(args.test_folder)
    # https://heartbeat.fritz.ai/building-a-data-pipeline-with-tensorflow-3047656b5095
    root_path = os.path.abspath(os.path.join("./dataset/mnt", "ramdisk/max/90kDICT32px"))
    with open(os.path.join(root_path, "annotation_train.txt"), "r") as fr:
        raw_data = fr.readlines()
    image_path_list = [os.path.join(root_path, re.match("./(.*.jpg)(.*)", image_path).group(1)) for image_path in raw_data]
    total_data_size = len(image_path_list)
    dataset = MJSynthDataset(image_path_list, (32, 100, 3))
    dataset = dataset.batch(args.batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    for epoch_idx in range(args.epochs):
        for batch_idx, data in enumerate(dataset):
            with tf.GradientTape() as tape:
                image_tensors, labels, paths = data
                decoded_labels = [[char.decode("utf-8") for char in np.asarray(label)] for label in labels]
                text, length = converter.encode(decoded_labels, batch_max_length=args.batch_max_length)

                if 'CTC' in args.Prediction:
                    raise NotImplementedError
                else:
                    trans, preds = net(image_tensors, text[:, :-1], is_train=False)  # align with Attention.forward
                    target = text[:, 1:]  # without [B] Symbol
                    losses = loss_fn(tf.one_hot(target, args.num_class), preds)

                    # save transformed image
                    # cv2.imwrite("in.jpg", np.asarray(image_tensors[0], np.int))
                    # cv2.imwrite("out.jpg", np.asarray(trans[0], np.int))

            gradients = tape.gradient(losses, net.trainable_variables)
            loss = tf.nn.compute_average_loss(losses)

            # training strategy
            #########################################
            # 1. Freeze parameters of Transformation net
            if loss > 5:
                trainable_variables = list()
                optimizer.apply_gradients(zip(gradients, net.trainable_variables[8:]))  # without Transformation

            # 2. Apply gradients to all net
            else:
                optimizer.apply_gradients(zip(gradients, net.trainable_variables))
            #########################################

            print("epoch_index: %d, batch_index: (%d, %d), loss: " % (epoch_idx, batch_idx, (total_data_size//args.batch_size) + (1 if total_data_size%args.batch_size > 0 else 0)), loss)
            # checkpoint.save(checkpoint_prefix)
            manager.save()

            # logs
            with summary_writer.as_default():
                tf.summary.scalar('loss', loss, step=epoch_idx)
                # tf.summary.trace_on(graph=True, profiler=True)
                # tf.summary.trace_export(name="func_trace", step=0, profiler_outdir=args.log_dir)

            batch_size = np.shape(data[0])[0]
            preds_index = tf.argmax(preds, axis=-1)
            length_for_pred = tf.zeros((batch_size, args.batch_max_length), dtype=tf.int32)
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

            with open(os.path.join(args.log_dir, log_filename), "a+") as fw:
                fw.write(log+'\n')
