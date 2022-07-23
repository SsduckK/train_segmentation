import os.path as op
from glob import glob
import cv2
import tensorflow as tf

import Config


def write_tfrecord():
    test_serializer()
    train_set, test_set = load_datset(Config.DATA_INFO.RAW_DATA_PATH, Config.DATA_INFO.RESOLUTION)
    make_tfrecord(train_set, "city", "train", Config.DATA_INFO.CLASS, Config.DATA_INFO.TFRECORD_PATH)
    make_tfrecord(train_set, "city", "train", Config.DATA_INFO.CLASS, Config.DATA_INFO.TFRECORD_PATH)


def test_serializer():
    pass


def load_cifar10_dataset(data_path, img_shape):
    pass


def read_data(file, img_shape):
    pass


def make_tfrecord(dataset, dataname, split, class_names, tfr_path):
    pass


def open_tfr_writer(writer, tfr_path, dataname, split, shard_index):
    pass


class TfrSerializer:
    def __call__(self, raw_example):
        features = self.convert_to_feature(raw_example)
        features = tf.train.Features(feature=features)
        tf_example = tf.train.Example(features=features)
        serialized = tf_example.SerializeToString()
        return serialized

    def convert_to_feature(self, raw_example):
        pass

    def _bytes_feature(value):
        pass

    def _float_feature(value):
        pass

    def _int64_feature(value):
        pass


if __name__ == "__main__":
    write_cifar10_tfrecord()

