import os.path as op

import keras.utils.vis_utils

import visualize_util as vu
import dataset_util as du
import VGG

import tensorflow as tf


dataset_root = '/media/cheetah/IntHDD/datasets/fcnn-dataset/dataset1'

train_image_path = op.join(dataset_root, 'images_prepped_train')
train_label_path = op.join(dataset_root, 'annotations_prepped_train')
test_image_path = op.join(dataset_root, 'images_prepped_test')
test_label_path = op.join(dataset_root, 'annotations_prepped_test')

training_image_paths, training_label_map_paths = du.get_datsset_slice_paths(train_image_path, train_label_path)
validation_image_paths, validation_label_map_paths = du.get_datsset_slice_paths(test_image_path, test_label_path)

training_dataset = du.get_validation_dataset(training_image_paths, training_label_map_paths)
validation_dataset = du.get_validation_dataset(validation_image_paths, validation_label_map_paths)


for class_name, color in zip(du.class_names, vu.colors):
    print(f"{class_name} -- {color}")

# vu.list_show_annotation(training_dataset)

model_fcn32, model_fcn16, model_fcn8 = VGG.segmentation_model()

sgd = tf.keras.optimizers.SGD(learning_rate=1e-2, momentum=0.9, nesterov=True)

model_fcn32.compile(loss='categorical_crossentropy',
                    optimizer=sgd,
                    metrics=['acc'])
model_fcn16.compile(loss='categorical_crossentropy',
                    optimizer=sgd,
                    metrics=['acc'])
model_fcn8.compile(loss='categorical_crossentropy',
                    optimizer=sgd,
                    metrics=['acc'])

train_count = len(training_image_paths)
valid_count = len(validation_image_paths)

EPOCHS = 170

steps_per_epoch = train_count//du.BATCH_SIZE
validation_steps = valid_count//du.BATCH_SIZE

model_fcn32.summary()
keras.utils.vis_utils.plot_model(model_fcn32)
history_fcn32 = model_fcn32.fit(training_dataset,
                                steps_per_epoch=steps_per_epoch,
                                validation_data=validation_dataset,
                                validation_steps=validation_steps,
                                epochs=100)