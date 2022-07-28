import os
import os.path as op
from glob import glob
import cv2
import numpy as np
import tensorflow as tf

import Config
import TFSerializer as tfs


def read_tfrecord():
    # gpu_config()
    train_dataset = get_dataset(Config.DATA_INFO.TFRECORD_PATH, "city", "train", True, Config.DATA_INFO.BATCH_SIZE)
    test_dataset = get_dataset(Config.DATA_INFO.TFRECORD_PATH, "city", "test", False, Config.DATA_INFO.BATCH_SIZE)
    check_data(train_dataset)
    check_data(test_dataset)


def get_dataset(tfr_path, dataname, split, shuffle=False, batch_size=4, epochs=1):
    tfr_files = tf.io.gfile.glob(op.join(tfr_path, f"{dataname}_{split}", "*.tfrecord"))
    tfr_files.sort()
    print("[TfrecordReader] tfr files:", tfr_files)
    dataset = tf.data.TFRecordDataset(tfr_files)
    dataset = dataset.map(parse_example)
    dataset = set_properties(dataset, shuffle, epochs, batch_size)
    return dataset


def parse_example(example):
    features = {
        "image": tf.io.FixedLenFeature(shape=(), dtype=tf.string, default_value=""),
        "mask": tf.io.FixedLenFeature(shape=(), dtype=tf.string, default_value=""),
    }
    parsed = tf.io.parse_single_example(example, features)
    parsed["image"] = tf.io.decode_raw(parsed["image"], tf.uint8)
    parsed["image_u8"] = tf.reshape(parsed["image"], Config.DATA_INFO.IMAGE_SHAPE)
    parsed["image"] = tf.image.convert_image_dtype(parsed["image_u8"], dtype=tf.float32)

    parsed["mask"] = tf.io.decode_raw(parsed["mask"], tf.uint8)
    parsed["mask_u8"] = tf.reshape(parsed["mask"], Config.DATA_INFO.IMAGE_SHAPE)
    parsed["mask"] = tf.image.convert_image_dtype(parsed["mask_u8"], dtype=tf.float32)
    return parsed


def set_properties(dataset, shuffle: bool, epochs: int, batch_size: int):
    if shuffle:
        dataset = dataset.shuffle(100)
    dataset = dataset.batch(batch_size).repeat(epochs)
    return dataset


def check_data(dataset):
    for i, features in enumerate(dataset):
        print("sample: ", i, features["image"].shape, features["mask"].shape)
        if i == 0:
            show_samples(features["image_u8"], features["mask_u8"])
        if i > 5:
            break


def show_samples(images, masks, grid=(3, 3)):
    for image, mask in zip(images, masks):
        print(image.shape)
        cv2.imshow("image", image.numpy())
        cv2.imshow("mask", mask.numpy())
        cv2.waitKey()


if __name__ == "__main__":
    read_tfrecord()
