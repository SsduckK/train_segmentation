import os
import os.path as op
from glob import glob
import cv2
import numpy as np
import tensorflow as tf
from time import time

import Config
import TFSerializer as tfs


def load_dataset(data_path):
    files = glob(op.join(data_path, '*.png'))
    train_image_sets = []
    train_mask_sets = []
    test_image_sets = []
    test_mask_sets = []
    for file in files:
        images, masks = read_data(file)
        if "train" in file:
            train_image_sets.append(images)
            train_mask_sets.append(masks)
        else:
            test_image_sets.append(images)
            test_mask_sets.append(masks)

    train_image_sets = np.array(train_image_sets)
    train_mask_sets = np.array(train_mask_sets)
    test_image_sets = np.array(test_image_sets)
    test_mask_sets = np.array(test_mask_sets)

    print("[load_cityscape_dataset] image_train", train_image_sets.shape, len(train_image_sets))
    print("[load_cityscape_dataset] mask_train", train_mask_sets.shape, len(train_mask_sets))

    return (train_image_sets, train_mask_sets), (test_image_sets, test_mask_sets)


def read_data(file):
    image = cv2.imread(file)
    mask = load_mask(file)
    return image, mask


def load_mask(file):
    mask_file = file.replace('leftImg8bit', 'gtFine')
    mask_file = mask_file[:-4] + '_color' + mask_file[-4:]
    mask = cv2.imread(mask_file)
    return mask


def make_tfrecord(dataset, dataname, split, tfr_path, dir_path):
    image, mask = dataset
    writer = None
    serializer = tfs.TfrSerializer()
    example_per_shard = 10000
    region = dir_path.split('/')[-1]
    for i, (img, msk) in enumerate(zip(image, mask)):
        if i % example_per_shard == 0:
            writer = open_tfr_writer(writer, tfr_path, dataname, split, i//example_per_shard, region)

        example = {"image": img, "mask": msk}
        serialized = serializer(example)
        writer.write(serialized)


def open_tfr_writer(writer, tfr_path, dataname, split, shard_index, dir_path):
    if writer:
        writer.close()

    tfrdata_path = op.join(tfr_path, f"{dataname}_{split}")
    if op.isdir(tfr_path) and not op.isdir(tfrdata_path):
        os.mkdir(tfrdata_path)
    tfrfile = op.join(tfrdata_path, f"shard_{dir_path}_{shard_index:03d}.tfrecord")
    writer = tf.io.TFRecordWriter(tfrfile)
    print(f"create tfrecord file: {tfrfile}")
    return writer


def write_tfrecord():
    dir_paths = glob(op.join(Config.DATA_INFO.RAW_DATA_PATH, "leftImg8bit", "leftImg8bit", '*', '*'))
    dir_paths.sort()
    for dir_path in dir_paths:
        start = time()
        train_set, test_set = load_dataset(dir_path)
        if not op.isdir(Config.DATA_INFO.TFRECORD_PATH):
            os.mkdir(Config.DATA_INFO.TFRECORD_PATH)
        make_tfrecord(train_set, "city", "train", Config.DATA_INFO.TFRECORD_PATH, dir_path)
        make_tfrecord(test_set, "city", "test", Config.DATA_INFO.TFRECORD_PATH, dir_path)
        end = time()
        print("time taken ", end - start)


if __name__ == "__main__":
    write_tfrecord()

