import keras
import keras.layers
import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras import Input
from tensorflow.python.keras.utils.vis_utils import plot_model
import tensorflow_addons as tfa
import numpy as np

import Config


def identity_block(X, f, filters, stage, block):
    conv_name_base = 'conv' + str(stage) + block
    bn_name_base = 'bn' + str(stage) + block

    F1, F2, F3 = filters

    shortcut = X

    X = tf.keras.layers.Conv2D(filters=F1, kernel_size=1, strides=1, padding='valid', name=conv_name_base + '2a',
                               kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0))(X)
    X = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = tf.keras.layers.Activation('relu')(X)

    X = tf.keras.layers.Conv2D(filters=F2, kernel_size=f, strides=1, padding='same', name=conv_name_base + '2b',
                               kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0))(X)
    X = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = tf.keras.layers.Activation('relu')(X)

    X = tf.keras.layers.Conv2D(filters=F3, kernel_size=1, strides=1, padding='valid', name=conv_name_base + '2c',
                               kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0))(X)
    X = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    X = tf.keras.layers.Add()([X, shortcut])
    X = tf.keras.layers.Activation('relu')(X)

    return X


def convolutional_block(X, f, filters, stage, block, s=2):
    conv_name_base = 'conv' + str(stage) + block
    bn_name_base = 'bn' + str(stage) + block

    F1, F2, F3 = filters

    shortcut = X

    X = tf.keras.layers.Conv2D(filters=F1, kernel_size=1, strides=s, padding='valid', name=conv_name_base + '2a',
                               kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0))(X)
    X = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = tf.keras.layers.Activation('relu')(X)

    X = tf.keras.layers.Conv2D(filters=F2, kernel_size=f, strides=1, padding='same', name=conv_name_base + '2b',
                               kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0))(X)
    X = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = tf.keras.layers.Activation('relu')(X)

    X = tf.keras.layers.Conv2D(filters=F3, kernel_size=1, strides=1, padding='valid', name=conv_name_base + '2c',
                               kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0))(X)
    X = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    X_shortcut = tf.keras.layers.Conv2D(filters=F3, kernel_size=1, strides=s, padding='valid',
                                        name=conv_name_base + '1',
                                        kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0))(shortcut)
    X_shortcut = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    X = tf.keras.layers.Add()([X, X_shortcut])
    X = tf.keras.layers.Activation('relu')(X)

    return X


def ResNet50(input_shape=Config.DATA_INFO.IMAGE_SHAPE, classes=len(Config.DATA_INFO.CLASS)):
    X_input = layers.Input(input_shape)

    X = layers.ZeroPadding2D((3, 3))(X_input)

    X = tf.keras.layers.Conv2D(filters=64, kernel_size=7, strides=2, name='conv1',
                               kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0))(X)
    X = tf.keras.layers.BatchNormalization(axis=3, name='bn_conv1')(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(X)

    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    X = tf.keras.layers.AveragePooling2D()(X)

    X = tf.keras.layers.Flatten()(X)
    X = tf.keras.layers.Dense(classes, activation='softmax', name='fc' + str(classes),
                              kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0))(X)

    model = tf.keras.models.Model(inputs=X_input, outputs=X, name='ResNet50')

    return model


model = ResNet50(input_shape=Config.DATA_INFO.IMAGE_SHAPE, classes=len(Config.DATA_INFO.CLASS))
model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metric=['acc'])