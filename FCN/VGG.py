import tensorflow as tf

vgg_weights_path = 'parameter/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'


def block(x, n_convs, filters, kernel_size, activation, pool_size, pool_stride, block_name):
    for i in range(n_convs):
        x = tf.keras.layers.Conv2D(filters=filters,
                                   kernel_size=kernel_size,
                                   activation=activation,
                                   padding='same',
                                   name=f'{block_name}_conv{i + 1}')(x)

    x = tf.keras.layers.MaxPooling2D(pool_size=pool_size,
                                     strides=pool_stride,
                                     name=f'{block_name}_pool{i + 1}')(x)

    return x


def VGG_16(image_input):
    x = block(image_input, n_convs=2, filters=64, kernel_size=(3, 3), activation='relu',
              pool_size=(2, 2), pool_stride=(2, 2),
              block_name='block1')
    p1 = x

    x = block(x, n_convs=2, filters=128, kernel_size=(3, 3), activation='relu',
              pool_size=(2, 2), pool_stride=(2, 2),
              block_name='block2')
    p2 = x

    x = block(x, n_convs=3, filters=256, kernel_size=(3, 3), activation='relu',
              pool_size=(2, 2), pool_stride=(2, 2),
              block_name='block3')
    p3 = x

    x = block(x, n_convs=3, filters=512, kernel_size=(3, 3), activation='relu',
              pool_size=(2, 2), pool_stride=(2, 2),
              block_name='block4')
    p4 = x

    x = block(x, n_convs=3, filters=512, kernel_size=(3, 3), activation='relu',
              pool_size=(2, 2), pool_stride=(2, 2),
              block_name='block5')
    p5 = x

    vgg = tf.keras.Model(image_input, p5)
    vgg.load_weights(vgg_weights_path)

    n = 4096

    c6 = tf.keras.layers.Conv2D(n, (7, 7), activation='relu', padding='same', name="conv6")(p5)
    c7 = tf.keras.layers.Conv2D(n, (1, 1), activation='relu', padding='same', name="conv7")(c6)

    return (p1, p2, p3, p4, c7)


def decoder(convs, n_classes):
    f1, f2, f3, f4, f5 = convs

    fcn32_o = tf.keras.layers.Conv2DTranspose(n_classes, kernel_size=(32, 32), strides=(32, 32), use_bias=False)(f5)
    fcn32_o = tf.keras.layers.Activation('softmax')(fcn32_o)

    o = tf.keras.layers.Conv2DTranspose(n_classes, kernel_size=(4, 4), strides=(2, 2), use_bias=False)(f5)
    o = tf.keras.layers.Cropping2D(cropping=(1, 1))(o)

    o2 = f4
    o2 = tf.keras.layers.Conv2D(n_classes, (1, 1), activation='relu', padding='same')(o2)

    o = tf.keras.layers.Add()([o, o2])  # (14, 14, n)

    fcn16_o = tf.keras.layers.Conv2DTranspose(n_classes, kernel_size=(16, 16), strides=(16, 16), use_bias=False)(o)
    fcn16_o = tf.keras.layers.Activation('softmax')(fcn16_o)

    o = tf.keras.layers.Conv2DTranspose(n_classes, kernel_size=(4, 4), strides=(2, 2), use_bias=False)(o)
    o = tf.keras.layers.Cropping2D(cropping=(1, 1))(o)

    o2 = f3
    o2 = tf.keras.layers.Conv2D(n_classes, (1, 1), activation='relu', padding='same')(o2)

    o = tf.keras.layers.Add()([o, o2])

    o = tf.keras.layers.Conv2DTranspose(n_classes, kernel_size=(8, 8), strides=(8, 8), use_bias=False)(o)
    fcn8_o = tf.keras.layers.Activation('softmax')(o)

    return fcn32_o, fcn16_o, fcn8_o


def segmentation_model():
    inputs = tf.keras.layers.Input(shape=(224, 224, 3, ))
    convs = VGG_16(inputs)
    fcn32, fcn16, fcn8 = decoder(convs, 12)
    model_fcn32 = tf.keras.Model(inputs, fcn32)
    model_fcn16 = tf.keras.Model(inputs, fcn16)
    model_fcn8 = tf.keras.Model(inputs, fcn8)

    return model_fcn32, model_fcn16, model_fcn8
