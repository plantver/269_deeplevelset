from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import collections

import tensorflow as tf
from archs.new_layers_2d import *
from data.augment import *

class Net():
    def _dice(self, i1, i2):
        with tf.variable_scope("dice"):
            return tf.reduce_sum(tf.multiply(i1, i2)) * 2 / (tf.reduce_sum(i1) + tf.reduce_sum(i2))

    def _binarydice(self, prob, target):
        with tf.variable_scope("binary_dice"):
            prob = tf.round(prob)
            return tf.reduce_sum(tf.multiply(prob, target)) * 2 / (tf.reduce_sum(prob) + tf.reduce_sum(target))

    def _convtranspose_concate(self, i1, i2, filters, padding='same', scope="convtranspose_concate", reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            return tf.concat(values=[
                i1,
                tf.layers.conv2d_transpose(i2, filters, 3, strides=(2,2), padding=padding)
            ], axis=-1)

    def __init__(self, config, patch, target, is_train_net=True, reuse=False, global_step=None):
        self.input = patch
        self.target = target
        self.name = config["name"]
        weight_decay = config["weight_decay"]
        learning_rate = config["learning_rate"]
        is_learning_rate_decay = config["is_learning_rate_decay"]

        self.summary_node_list = list()
        self.stream_update_ops = list()
        self.end_points = collections.OrderedDict()

        # data preprocess and augmentation
        with tf.variable_scope("augmentation"):
            self.input = tf.expand_dims(self.input, -1)
            # self.target = tf.expand_dims(self.target, -1)
            # self.input_shape = self.input.shape.as_list()
            self.input = scaling(self.input, -800, 800)
            # if is_train_net:
            #     pass
            #     # self.input = rand_transpose_rotate_3D_Z(self.input)
            # self.input = tf.slice(self.input, tf.convert_to_tensor([0, 10, 10, 10, 0]),
            #                       tf.convert_to_tensor([self.input_shape[0], 81, 81, 81, 1]))

        # convnet
        with tf.variable_scope(self.name, reuse=reuse):

            layer_name = 'conv1'
            net = self.end_points[layer_name] = add_conv_bn_relu(self.input, 16, 3, is_train_net, layer_name,
                                                                 weight_decay)

            layer_name = 'pool1'
            net = self.end_points[layer_name] = tf.layers.max_pooling2d(net, 2, 2, padding='same')

            layer_name = 'conv2'
            net = self.end_points[layer_name] = add_conv_bn_relu(net, 32, 3, is_train_net, layer_name,
                                                                 weight_decay)

            layer_name = 'pool2'
            net = self.end_points[layer_name] = tf.layers.max_pooling2d(net, 2, 2, padding='same')

            layer_name = 'conv3'
            net = self.end_points[layer_name] = add_conv_bn_relu(net, 64, 3, is_train_net, layer_name,
                                                                 weight_decay)

            layer_name = 'pool3'
            net = self.end_points[layer_name] = tf.layers.max_pooling2d(net, 2, 2, padding='same')

            layer_name = 'conv4'
            net = self.end_points[layer_name] = add_conv_bn_relu(net, 128, 3, is_train_net, layer_name,
                                                                 weight_decay)

            layer_name = 'pool4'
            net = self.end_points[layer_name] = tf.layers.max_pooling2d(net, 2, 2, padding='same')

            layer_name = 'conv5'
            net = self.end_points[layer_name] = add_conv_bn_relu(net, 256, 3, is_train_net, layer_name,
                                                                 weight_decay)

            # ==== upward ====
            layer_name = 'up_concate1'
            net = self.end_points[layer_name] = self._convtranspose_concate(
                self.end_points["conv4"], self.end_points["conv5"], 128, scope=layer_name
            )

            layer_name = 'conv6'
            net = self.end_points[layer_name] = add_conv_bn_relu(net, 128, 3, is_train_net, layer_name,
                                                                 weight_decay)

            layer_name = 'up_concate2'
            net = self.end_points[layer_name] = self._convtranspose_concate(
                self.end_points["conv3"], self.end_points["conv6"], 64, scope=layer_name
            )

            layer_name = 'conv7'
            net = self.end_points[layer_name] = add_conv_bn_relu(net, 64, 3, is_train_net, layer_name,
                                                                 weight_decay)

            layer_name = 'up_concate3'
            net = self.end_points[layer_name] = self._convtranspose_concate(
                self.end_points["conv2"], self.end_points["conv7"], 32, scope=layer_name
            )

            layer_name = 'conv8'
            net = self.end_points[layer_name] = add_conv_bn_relu(net, 32, 3, is_train_net, layer_name,
                                                                 weight_decay)

            layer_name = 'up_concate4'
            net = self.end_points[layer_name] = self._convtranspose_concate(
                self.end_points["conv1"], self.end_points["conv8"], 16, scope=layer_name
            )

            layer_name = 'conv9'
            net = self.end_points[layer_name] = add_conv_bn_relu(net, 16, 3, is_train_net, layer_name,
                                                                 weight_decay)

            layer_name = 'conv10'
            self.logits = self.end_points[layer_name] = add_conv_bn_relu(net, 2, 3, is_train_net, layer_name,
                                                                 weight_decay)

        # model outputs
        with tf.variable_scope("model_outpus"):
            self.probabilities = tf.nn.sigmoid(self.logits)

        # objective
        with tf.variable_scope("losses"):
            with tf.variable_scope("regularization"):
                self.reg_losses = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
                self.reg_losses = tf.Print(self.reg_losses, [self.reg_losses], message='reg_losses: ')
            with tf.variable_scope("softmax_cross_entropy"):
                self.loss_sce = tf.reduce_sum(tf.losses.sigmoid_cross_entropy(self.target, self.logits))
            self.loss = tf.add(self.loss_sce, self.reg_losses)
            self.dice = self._dice(self.probabilities, self.target)
            self.binary_dice = self._binarydice(self.probabilities, self.target)
            self.loss = tf.add(self.loss, -self.binary_dice)
            self.binary_mask = tf.round(self.probabilities)

        # optimizer
        if is_train_net:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):  # to ensure that the global stats of BN layers are updated
                with tf.variable_scope("Optimizer"):
                    if is_learning_rate_decay:
                        learning_rate_expdecay = tf.train.exponential_decay(learning_rate, global_step,
                                                                   2000, 0.96, staircase=True)
                    # self.optimizer = tf.train.MomentumOptimizer(
                    #     learning_rate=learning_rate_expdecay, momentum=0.99, use_nesterov=True
                    # ).minimize(self.loss, global_step=global_step)
                    self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss, global_step=global_step)

        # summaries
        prefix = "train_" if is_train_net else "validation_"
        with tf.variable_scope(prefix + "summaries"):
            # self.acc, self.acc_update_op = tf.contrib.metrics.streaming_accuracy(self.prediction, self.label)
            # self.stream_update_ops.append(self.acc_update_op)
            # self.summary_node_list.append(tf.summary.scalar(prefix + "streaming_accuracy", self.acc))
            # self.auc, self.auc_update_op = tf.contrib.metrics.streaming_auc(self.probabilities, self.label)
            # self.stream_update_ops.append(self.auc_update_op)
            # self.summary_node_list.append(tf.summary.scalar(prefix + 'streaming_auc', self.auc))
            self.summary_node_list.append(tf.summary.scalar(prefix + 'batch_loss', self.loss_sce))
            self.summary_node_list.append(tf.summary.scalar(prefix + 'dice', self.dice))
            # self.summary_node_list.append(tf.summary.scalar(prefix + 'cross-entropy', self.loss_sce))
            self.summary_node_list.append(tf.summary.scalar(prefix + 'binary-dice', self.binary_dice))
            self.summary_node_list.append(tf.summary.image(prefix + 'img',
                                                           tf.concat([self.input[:, :, :, :],
                                                                      tf.expand_dims(self.target[ :, :, :, 0], axis=-1),
                                                                      tf.expand_dims(self.target[:, :, :, 1], axis=-1),
                                                                      tf.expand_dims(self.probabilities[:, :, :, 0], axis=-1),
                                                                      tf.expand_dims(self.probabilities[:, :, :, 1], axis=-1)], axis=2),
                                                           max_outputs=20))

            if is_train_net:
                self.summary_node_list.append(tf.summary.scalar(prefix + 'global_step', global_step))
                self.summary_node_list.append(tf.summary.scalar(prefix + 'learing_rate', learning_rate_expdecay))

            # histogram
            net_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
            for v in net_vars:
                if "bn" in v.name:
                    self.summary_node_list.append(tf.summary.histogram(v.name, v))

            for end_point in self.end_points:
                self.summary_node_list.append(tf.summary.histogram("activation/" + end_point, self.end_points[end_point]))

        self.print_base_graph()

    def print_base_graph(self):
        for layer in self.end_points:
            print('{:10s}:{}'.format(layer, self.end_points[layer].shape))

    def load_pretrained(self, sess, pretrained_model):
        vars_to_load = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        io = tf.train.Saver(vars_to_load)
        io.restore(sess, pretrained_model)
