import tensorflow as tf
import numpy as np
import threading
import h5py
from data.augment import balance_classes, affineRotAroundCenter



class InputQueue():
    def __init__(self, config,):
        self.config = config

        print("...loading h5 files")
        self.dct_load = dict()
        self.lst_h5fs = [h5py.File(x, 'r') for x in self.config["h5files"]]
        for h5f in self.lst_h5fs:
            self.dct_load.update(h5f)

        if self.config["balance"]["flg"]:
            self.lst_sampleIds = balance_classes(self.config["balance"]["ratio"],
                                                 self.dct_load,
                                                 self.config["balance"]["class_key"])
        else:
            self.lst_sampleIds = list(self.dct_load.keys())
        np.random.shuffle(self.lst_sampleIds)
        print("total samples: ", len(self.lst_sampleIds))

        # create queue according to load
        self.batch_size = self.config["batch_size"]
        capacity = 3 * self.batch_size
        with tf.variable_scope("InputQueue"):
            self.queue = tf.RandomShuffleQueue(capacity=capacity,
                                    min_after_dequeue=int(0.9*capacity),
                                    shapes=[x["shape"] for x in self.config["load"]], 
                                    dtypes=[tf.float32] * len(self.config["load"]))

            self.lst_placeholders = [
                tf.placeholder(tf.float32, shape=[self.batch_size,] + x["shape"], name="InputPlaceholder/%s"%(x["name"]))
                for x in self.config["load"]]
            self.enqueue_op = self.queue.enqueue_many(self.lst_placeholders)
            self.dequeue_op = self.queue.dequeue_many(self.batch_size)

    def _fill_queue(self,):
        under = 0
        MAX = len(self.lst_sampleIds)
        batch_size = self.batch_size
        while not self.coord.should_stop():
            upper = under + batch_size
            if upper <= MAX:
                lst_id = self.lst_sampleIds[under:upper]
                under = upper
            else:
                rest = upper - MAX
                lst_id = self.lst_sampleIds[under:MAX]+self.lst_sampleIds[0:rest]
                under = rest
            curr_data = [[self.dct_load[i][x["name"]][()] for i in lst_id]
                for x in self.config["load"]]

            try:
                self.sess.run(self.enqueue_op, feed_dict={t[0]: t[1] for t in zip(self.lst_placeholders, curr_data)})
            except tf.errors.CancelledError:
                return

        for h5f in self.lst_h5fs:
            h5f.close()

    def start_filling(self, sess, coord):
        self.sess = sess
        self.coord = coord
        threading.Thread(target=self._fill_queue).start()

    def stop(self,):
        self.coord.request_stop()
        for h5f in self.lst_h5fs:
            h5f.close()



class simple_generator():
    def __init__(self, config):
        self.config = config
        self.dct_load = dict()
        self.lst_h5fs = [h5py.File(x, 'r') for x in config["h5files"]]
        for h5f in self.lst_h5fs:
            self.dct_load.update(h5f)
        self.lst_sampleIds = list(self.dct_load.keys())
        print("total input %s"%(len(self.lst_sampleIds),))

        with tf.variable_scope("simple_generator"):
            self.lst_placeholders = [
                tf.placeholder(tf.float32, shape=[1] + x["shape"],
                               name="InputPlaceholder/%s" % (x["name"]))
                for x in self.config["load"]]

    def generate_feed_dict(self):
        for sampleId in self.lst_sampleIds:
            yield {t[0]: np.expand_dims(self.dct_load[sampleId][t[1]["name"]][()], axis=0)
                   for t in zip(self.lst_placeholders, self.config["load"])}, sampleId

        for h5f in self.lst_h5fs:
            h5f.close()



if __name__ == "__main__":

    config = {
        "h5files": [
            "/mnt/dfs/xjyan/patches/NLST_clas_valid_side101_spac0.5_noaug_filtered.h5",
            ],
        "batch_size": 5,
        "load": [
            {
                "name": "patch",
                "shape": [101,101,101]
            },
            {
                "name": "class",
                "shape": []
            }
        ]
    }

    sess = tf.Session()
    coord = tf.train.Coordinator()

    IQ = InputQueue(config)
    IQ.start_filling(sess, coord)
    deq = IQ.dequeue_op
    for i in range(4):
        print("dequeueueueueu", [x.shape for x in sess.run(deq)])


    IQ.stop()

