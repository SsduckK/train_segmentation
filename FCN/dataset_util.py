import os
import os.path as op
import numpy as np

import tensorflow as tf

class_names = ['sky', 'building','column/pole', 'road', 'side walk', 'vegetation', 'traffic light', 'fence', 'vehicle', 'pedestrian', 'bicyclist', 'void']
BATCH_SIZE = 2

def map_filename_to_image_and_mask(t_filename, a_filename, height=224, width=224):
    img_raw = tf.io.read_file(t_filename)
    anno_raw = tf.io.read_file(a_filename)
    image = tf.image.decode_jpeg(img_raw)
    annotation = tf.image.decode_jpeg(anno_raw)

    image = tf.image.resize(image, (height, width,))
    annotation = tf.image.resize(annotation, (height, width,))
    image = tf.reshape(image, (height, width, 3))
    annotation = tf.cast(annotation, dtype=tf.int32)
    annotation = tf.reshape(annotation, (height, width, 1, ))
    stack_list = []

    for c in range(len(class_names)):
        mask = tf.equal(annotation[:, :, 0], tf.constant(c))
        stack_list.append(tf.cast(mask, dtype=tf.int32))

    annotation = tf.stack(stack_list, axis=2)

    image = image / 127.5
    image -= 1

    return image, annotation


def get_datsset_slice_paths(image_dir, label_map_dir):
    image_file_list = os.listdir(image_dir)
    label_map_file_list = os.listdir(label_map_dir)
    image_paths = [op.join(image_dir, fname) for fname in image_file_list]
    label_map_paths = [op.join(label_map_dir, fname) for fname in label_map_file_list]

    return image_paths, label_map_paths


def get_validation_dataset(image_paths, label_map_paths):
    validation_dataset = tf.data.Dataset.from_tensor_slices((image_paths, label_map_paths))
    validation_dataset = validation_dataset.map(map_filename_to_image_and_mask)
    validation_dataset = validation_dataset.batch(BATCH_SIZE)
    validation_dataset = validation_dataset.repeat()

    return validation_dataset