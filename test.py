from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import os
import re
import importlib
import numpy as np
import json
import argparse
from shutil import copyfile
import tensorflow as tf
from data.inputQueue import simple_generator
from tqdm import tqdm
import h5py
#import pdb; pdb.set_trace()
"""
example usage: python train.py --cfg cfgs/xxx.cfg
"""

def _get_iter(s):
    return int(re.search(r'[0-9]+$', s).group(0))

def main(config_file):
    # ============================params
    params = json.load(open(config_file, 'rt'))["test"]
    parent_save_location = params['parent_save_location']
    project_tag = params['project_tag']
    net_name = params['net_name']
    pretrained_model = params['pretrained_model']
    gpu_id = params['gpu_id']
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    # ============================params

    # create test directory
    project_dir = os.path.join(parent_save_location, project_tag)
    testresult_dir = os.path.join(project_dir, "test")
    if not os.path.exists(testresult_dir):
        os.makedirs(testresult_dir)

    # copy config file to the target dir
    copyfile(config_file, os.path.join(testresult_dir, 'test_setting.cfg'))

    # ================== Get net graph
    input_gen = simple_generator(params["validation_input"])

    net_class = importlib.import_module("archs." + net_name).Net
    net = net_class(params["net"], *input_gen.lst_placeholders, is_train_net=False, reuse=False)

    # ================== TEST
    # create output h5 file
    h5f = h5py.File(os.path.join(testresult_dir, "%s_predictions.h5" % (params["prefix"])), "w")

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

        for feed_dict, pid in tqdm(input_gen.generate_feed_dict()):
            pmap, dice = sess.run([net.probabilities, net.dice], feed_dict=feed_dict)
            g = h5f.create_group(pid)
            g.create_dataset("pmap", data=pmap)
            g.create_dataset("dice", data=dice)

    # save the predictions
    h5f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg')
    args = parser.parse_args()
    main(args.cfg)
