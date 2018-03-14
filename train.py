from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import os
import re
import shutil
import json
import argparse
import importlib
from shutil import copyfile
import tensorflow as tf
from data.inputQueue import InputQueue
# import pdb; pdb.set_trace()
"""
example usage: python train.py --cfg cfgs/xxx.cfg
"""


def _get_iter(s):
    """
    TODO: use gloable variable to save global_step count
    """
    return int(re.search(r'[0-9]+$', s).group(0))


def main(config_file):
    # ============================params
    params = json.load(open(config_file, 'rt'))["train"]
    parent_save_location = params['parent_save_location']
    project_tag = params['project_tag']
    net_name = params['net_name']
    train_summary_interval = params['train_summary_interval']
    validation_interval = params['validation_interval']
    snapshot_interval = params['snapshot_interval']
    max_global_step = params['max_iteration']
    pretrained_model = params['pretrained_model']
    gpu_id = params['gpu_id']
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    # ============================params

    # create experiment directory
    project_dir = os.path.join(parent_save_location, project_tag)
    snapshot_dir = os.path.join(project_dir, 'snapshots')
    log_dir = os.path.join(project_dir, 'logs')
    if not os.path.exists(snapshot_dir):
        os.makedirs(snapshot_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # else:
    #     shutil.rmtree(log_dir)
    #     os.makedirs(log_dir)

    # copy config file to the target dir
    copyfile(config_file, os.path.join(project_dir, os.path.basename(config_file)))

    # global step
    with tf.variable_scope("Global_step"):
        global_step = tf.Variable(0, name="global_step", trainable=False)
    def get_glbstep(sess):
        return tf.train.global_step(sess, global_step)


    # ================== TRAIN
    # create train data pipeline
    iq_train = InputQueue(params["input"])
    input_tensors = iq_train.dequeue_op

    # create base train net
    net_class = importlib.import_module("archs." + net_name).Net
    tr_net = net_class(params["net"], *input_tensors, is_train_net=True, reuse=False, global_step=global_step)

    # set up tr summary writer
    tr_summary_writer = tf.summary.FileWriter(log_dir + '/train', flush_secs=30)
    tr_summaries = tf.summary.merge(tr_net.summary_node_list)
    tr_summary_writer.add_graph(tf.get_default_graph())

    # ================== VALIDATION
    # iq_valid = InputQueue(params["validation_input"])
    # valid_tensors = iq_valid.dequeue_op
    # val_net = net_class(params["net"], *valid_tensors, is_train_net=False, reuse=True, global_step=global_step)
    # val_summary_writer = tf.summary.FileWriter(log_dir + '/val', flush_secs=30)
    # val_summaries = tf.summary.merge(val_net.summary_node_list)
    # val_summary_writer.add_graph(tf.get_default_graph())

    # ================== Training loop
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)
    coord = tf.train.Coordinator()
    with tf.Session(config=sess_config) as sess:

        # start filling the input Queue
        iq_train.start_filling(sess, coord)
        # iq_valid.start_filling(sess, coord)

        # init variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        # load a pre-trained model
        if pretrained_model is not None:
            print("LOADING CKPT %s"%(pretrained_model))
            saver.restore(sess, pretrained_model)
        else:
            latest_model = tf.train.latest_checkpoint(snapshot_dir)
            print("LOADING LATEST CKPT %s"%(latest_model))
            if latest_model is not None:
                saver.restore(sess, latest_model)

        while get_glbstep(sess) < max_global_step:
            sess.run([tr_net.optimizer, tr_net.stream_update_ops])

            print('step %s dice %s binary dice %s'%(
                get_glbstep(sess), sess.run(tr_net.dice), sess.run(tr_net.binary_dice)))

            if get_glbstep(sess) % train_summary_interval == 0:
                summaries_evaluated = sess.run(tr_summaries)
                tr_summary_writer.add_summary(summaries_evaluated, get_glbstep(sess))

            # if get_glbstep(sess) % validation_interval == 0:
            #     sess.run(val_net.stream_update_ops)
            #     summaries_evaluated = sess.run(val_summaries)
            #     val_summary_writer.add_summary(summaries_evaluated, get_glbstep(sess))

            if get_glbstep(sess) % snapshot_interval == 0:
                saver.save(sess, os.path.join(snapshot_dir, project_tag + '_iter_' + str(get_glbstep(sess))))

        saver.save(sess, os.path.join(snapshot_dir, project_tag + '_iter_' + str(get_glbstep(sess))))

    iq_train.stop()
    # iq_valid.stop()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg')
    args = parser.parse_args()
    main(args.cfg)
