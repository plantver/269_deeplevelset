import tensorflow as tf
import numpy as np


l2_reg = tf.contrib.layers.l2_regularizer


def gap_3D(input, name="global_average_pooling_3D"):
    """
    global average pooling
    :param input: 5D tensor of shape [N, D, H, W, C]
    :return: 2D tensor [N, C]
    """
    return tf.reduce_mean(input, axis=[1, 2, 3], name=name)


def maxpool_center_concat_crop_2_3D(input, name="maxpool_center_crop", reuse=None):
    _shape = np.array(input.shape.as_list())
    _size = list(map(int, [_shape[0]] + np.floor(_shape[1:-1] / 2).tolist() + [_shape[-1]]))
    _begin = list(map(int, [0] + np.floor((_shape[1:-1] - _size[1:-1]) / 2).tolist() + [0]))
    with tf.variable_scope(name):
        return tf.concat(
            [tf.layers.max_pooling3d(input, 2, 2, padding='valid'),
             tf.slice(input, begin=_begin, size=_size)],
            axis=-1
        )


def add_conv_bn_sigmoid(model, kernels, ker_size, is_training, layer_name, weight_decay=None):
    if weight_decay:
        model = tf.layers.conv3d(model, kernels, ker_size, kernel_regularizer=l2_reg(scale=weight_decay),
                                 name=layer_name, padding='same',
                                 use_bias=True)  # kernel_initializer by default is xavier
    else:
        print('ok')
        model = tf.layers.conv3d(model, kernels, ker_size, name=layer_name, padding='same',
                                 use_bias=True)  # kernel_initializer by default is xavier

    model = tf.layers.batch_normalization(model, center=True, scale=True, training=is_training, name=layer_name + '/bn')
    model = tf.nn.sigmoid(model, name=layer_name + '/sigmoid')

    return model


def add_conv_bn_relu(model, kernels, ker_size, is_training, layer_name, weight_decay=None, alpha=0.01):
    if weight_decay:
        model = tf.layers.conv3d(model, kernels, ker_size, kernel_regularizer=l2_reg(scale=weight_decay),
                                 name=layer_name, padding='same',
                                 use_bias=True)  # kernel_initializer by default is xavier
    else:
        model = tf.layers.conv3d(model, kernels, ker_size, name=layer_name, padding='same',
                                 use_bias=True)  # kernel_initializer by default is xavier

    model = tf.layers.batch_normalization(model, center=True, scale=True, training=is_training, name=layer_name + '/bn')
    model = tf.nn.leaky_relu(model, alpha=alpha, name=layer_name + '/relu')

    return model


def incep_res_concat_bn_relu_drop_3D(net,
                                     num_filters,
                                     kernel_size,
                                     is_train_net,
                                     layer_name,
                                     scale=1.0,
                                     weight_decay=None,
                                     reuse=None,
                                     dropout=0.5,
                                     alpha=0.01):
    """
    @ https://github.com/tensorflow/nets/blob/master/research/slim/nets/inception_resnet_v2.py
    Builds the 35x35 resnet block.
    """

    with tf.variable_scope(layer_name, reuse=reuse):
        kernel_regularizer = tf.contrib.layers.l2_regularizer(weight_decay) if weight_decay else None

        with tf.variable_scope('Branch_0'):
            tower_conv = tf.layers.conv3d(net, num_filters, 1,
                                          kernel_regularizer=kernel_regularizer,
                                          name="Conv3d_1x1", padding='same',
                                          use_bias=True)  # kernel_initializer by default is xavier
        with tf.variable_scope('Branch_1'):
            tower_conv1_0 = tf.layers.conv3d(net, num_filters, 1,
                                             kernel_regularizer=kernel_regularizer,
                                             name="Conv3d_0a_1x1", padding='same', use_bias=True)
            tower_conv1_1 = tf.layers.conv3d(tower_conv1_0, num_filters, kernel_size,
                                             kernel_regularizer=kernel_regularizer,
                                             name="Conv3d_0b_3x3", padding='same', use_bias=True)
        with tf.variable_scope('Branch_2'):
            tower_conv2_0 = tf.layers.conv3d(net, num_filters, 1,
                                             kernel_regularizer=kernel_regularizer,
                                             name="Conv3d_0a_1x1", padding='same', use_bias=True)
            tower_conv2_1 = tf.layers.conv3d(tower_conv2_0, num_filters, kernel_size,
                                             kernel_regularizer=kernel_regularizer,
                                             name="Conv3d_0b_3x3", padding='same', use_bias=True)
            tower_conv2_2 = tf.layers.conv3d(tower_conv2_1, num_filters, kernel_size,
                                             kernel_regularizer=kernel_regularizer,
                                             name="Conv3d_0c_3x3", padding='same', use_bias=True)
        mixed = tf.concat(axis=-1, values=[tower_conv, tower_conv1_1, tower_conv2_2])
        mixed = tf.layers.conv3d(mixed, net.get_shape()[-1], 1,
                                 kernel_regularizer=kernel_regularizer,
                                 name="Conv3d_1x1", padding='same', use_bias=True)
        # scaled_up = mixed * scale
        # if activation_fn == tf.nn.relu6:
        #     # Use clip_by_value to simulate bandpass activation.
        #     scaled_up = tf.clip_by_value(scaled_up, -6.0, 6.0)

        net += mixed
        net = tf.layers.batch_normalization(net, training=is_train_net)
        net = tf.nn.leaky_relu(net, alpha=alpha)
        if dropout:
            net = tf.layers.dropout(net, dropout, training=is_train_net)
    return net


def incep_pool_concat_bn_relu_drop_3D(net,
                                      num_filters,
                                      kernel_size,
                                      is_train_net,
                                      layer_name,
                                      weight_decay=None,
                                      reuse=None,
                                      dropout=0.5,
                                      alpha=0.01):
    with tf.variable_scope(layer_name, reuse=reuse):
        kernel_regularizer = tf.contrib.layers.l2_regularizer(weight_decay) if weight_decay else None

        with tf.variable_scope('Branch_0'):
            tower_conv = tf.layers.conv3d(net, num_filters, 1, strides=2,
                                          kernel_regularizer=kernel_regularizer,
                                          name="Conv3d_1x1", padding='same', use_bias=True)
        with tf.variable_scope('Branch_1'):
            tower_conv1_0 = tf.layers.conv3d(net, num_filters, 1,
                                             kernel_regularizer=kernel_regularizer,
                                             name="Conv3d_0a_1x1", padding='same', use_bias=True)
            tower_conv1_1 = tf.layers.conv3d(tower_conv1_0, num_filters, kernel_size, strides=2,
                                             kernel_regularizer=kernel_regularizer,
                                             name="Conv3d_0b_3x3", padding='same', use_bias=True)
        with tf.variable_scope('Branch_2'):
            tower_pool = tf.layers.average_pooling3d(net, 2, 2, padding="same",
                                                     name="Pool")
        mixed = tf.concat(axis=-1, values=[tower_conv, tower_conv1_1, tower_pool])
        mixed = tf.layers.conv3d(mixed, mixed.get_shape()[-1], 1,
                                 kernel_regularizer=kernel_regularizer,
                                 name="Conv3d_1x1", padding='same', use_bias=True)

        mixed = tf.layers.batch_normalization(mixed, training=is_train_net)
        net = tf.nn.leaky_relu(mixed, alpha=alpha)
        if dropout:
            net = tf.layers.dropout(net, dropout, training=is_train_net)
    return net
