from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import os
import re
import importlib
import numpy as np
import json
import tensorflow as tf
from data.inputQueue import simple_generator
import h5py
from msnake import MSnake
import matplotlib.pyplot as plt

def _stich(im1, im0):
    im = np.zeros(im1.shape)
    im[:, :256] = im1[:, :256]
    im[:, 256:] = im0[:, 256:]
    return im


def _normalize(img):
    ma = np.max(img)
    mi = np.min(img)
    quo = ma - mi
    if quo == 0:
        return np.zeros(img.shape)
    return (img - mi)/(ma-mi)


def _dice(i1, i2):
    i1 = _normalize(i1)
    i2 = _normalize(i2)
    quo = (np.sum(i1) + np.sum(i2))
    if quo == 0:
        return 0

    return 2 * np.sum(i1 * i2) / quo


def _imshow(img, title):
    plt.figure(dpi=300)
    plt.imshow(img)
    plt.title(title)
    plt.show(block=False)


def main(params):
    # ============================params
    net_name = params['net_name']
    pretrained_model = params['pretrained_model']
    gpu_id = params['gpu_id']
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    # ============================params

    # ================== Get net graph
    input_gen = simple_generator(params["validation_input"])

    net_class = importlib.import_module("archs." + net_name).Net
    net = net_class(params["net"], *input_gen.lst_placeholders, is_train_net=False, reuse=False)

    # ================== TEST
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())  # for the layers not in pre-trained model

        # load a pre-trained model
        if pretrained_model is not None:
            print("Loading model: %s"%(pretrained_model,))
            saver = tf.train.Saver()
            saver.restore(sess, pretrained_model)
        else:
            print("NO TRAINED MODEL!")
            exit(1)

        for feed_dict, pid in input_gen.generate_feed_dict():
            input, binary_mask, binary_dice, target = sess.run([net.input, net.binary_mask, net.binary_dice, net.target], feed_dict=feed_dict)
            break

        # _imshow(input[0,:,:,0], "Input Image")
        # _imshow(_stich(binary_mask[0,:,:,1], binary_mask[0,:,:,0]), "FCN segmentation mask")

        print("\n FCN Binary Dice Against Ground Truth: ", binary_dice)

        print("\n Start MGAC iteration for left lung ...")
        msnake_0 = MSnake(binary_mask[0,:,:,0], input[0,:,:,0], iterations=50)
        _ = msnake_0.evolve()
        mmask_0 = msnake_0.mask

        print("\n Start MGAC iteration for right lung ...")
        msnake_1 = MSnake(binary_mask[0, :, :, 1], input[0, :, :, 0], iterations=50)
        _ = msnake_1.evolve()
        mmask_1 = msnake_1.mask

        # _imshow(_stich(mmask_1, mmask_0), "MGAC segmentation mask")

        print("\n MGAC Binary Dice: ", _dice(_stich(target[0,:,:,1], target[0,:,:,0]), _stich(mmask_1, mmask_0)))

        plt.figure(dpi=200)
        plt.subplot(1, 3, 1)
        plt.imshow(input[0,:,:,0])
        plt.title("Input Image")
        plt.subplot(1, 3, 2)
        plt.imshow(_stich(binary_mask[0,:,:,1], binary_mask[0,:,:,0]))
        plt.title("FCN")
        plt.subplot(1, 3, 3)
        plt.imshow(_stich(mmask_1, mmask_0))
        plt.title("MGAC")
        plt.show()

if __name__ == '__main__':
    PATH_demo_cfg = "demo/demo.cfg.json"

    main(json.load(open(PATH_demo_cfg, 'rt')))
