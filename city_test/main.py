import os.path as op
from glob import glob
import cv2
import numpy as np
import tensorflow as tf

import Config


def test_serializer():
    pass


def load_dataset(data_path, img_shape):
    files = glob(op.join(data_path, 'leftImg8bit', 'leftImg8bit', '*', '*', '*'))
    train_image = []
    train_mask = []

    test_image = []
    test_mask = []
    for file in files:
        images, masks = read_data(file, img_shape)
        if "train" in file:
            train_image.append(images)
            train_mask.append(masks)
        else:
            test_image.append(images)
            test_mask.append(masks)
    train_image = np.concatenate(train_image, axis=0)
    test_image = np.concatenate(test_image, axis=0)

    print("load cityscape train image/mask shape: ", train_image.shape, len(train_image))
    print("load cityscape test image/mask shape: ", train_image.shape, len(test_image))


def read_data(file, img_shape):
    image = cv2.imread(file)
    mask = load_mask(file)
    return image, mask


def load_mask(file):
    mask_file = file.replace('leftImg8bit', 'gtFine')
    mask_file = mask_file[:-4] + '_color' + mask_file[-4:]
    mask = cv2.imread(mask_file)
    return mask


def make_tfrecord(dataset, dataname, split, class_names, tfr_path):
    pass


def open_tfr_writer(writer, tfr_path, dataname, split, shard_index):
    pass


def write_tfrecord():
    test_serializer()
    train_set, test_set = load_dataset(Config.DATA_INFO.RAW_DATA_PATH, Config.DATA_INFO.IMAGE_SHAPE)
    make_tfrecord(train_set, "city", "train", Config.DATA_INFO.CLASS, Config.DATA_INFO.TFRECORD_PATH)
    make_tfrecord(train_set, "city", "train", Config.DATA_INFO.CLASS, Config.DATA_INFO.TFRECORD_PATH)


if __name__ == "__main__":
    write_tfrecord()

