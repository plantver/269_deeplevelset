import tensorflow as tf
import numpy as np

l2_reg = tf.contrib.layers.l2_regularizer


def gap_2D(input, name="global_average_pooling_3D"):
    """
    global average pooling
    :param input: 5D tensor of shape [N, H, W, C]
    :return: 2D tensor [N, C]
    """
    return tf.reduce_mean(input, axis=[1, 2], name=name)


def add_conv_bn_relu(model, kernels, ker_size, is_training, layer_name, weight_decay=None, alpha=0.01):
    kernel_regularizer = l2_reg(scale=weight_decay) if weight_decay else None
    model = tf.layers.conv2d(model, kernels, ker_size,
                             kernel_regularizer=kernel_regularizer,
                             name=layer_name, padding='same',
                             use_bias=True)  # kernel_initializer by default is xavier

    model = tf.layers.batch_normalization(model, center=True, scale=True, training=is_training, name=layer_name + '/bn')
    model = tf.nn.leaky_relu(model, alpha=alpha, name=layer_name + '/leaky_relu')

    return model


def z_sliding_maxfilters(model):
    """
    input a 5D tensor [N, D, H, W, C]
    apply 2D conv along dimension D, followed by maxpool

    :param model:
    :return:
    """

    pass

