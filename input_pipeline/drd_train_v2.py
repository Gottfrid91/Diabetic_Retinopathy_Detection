"""A binary to train CIFAR-10 using a single GPU.
Accuracy:
cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py.
Speed: With batch_size 128.
System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)
Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.
http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from matplotlib.pyplot import imshow
from matplotlib import pyplot as plt
from PIL import Image

from datetime import datetime
import os.path
import time
import unicodedata


import numpy as np

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import sys
import drd

#write no pyc files
sys.dont_write_bytecode = True

parser = drd.parser


parser.add_argument('--max_steps', type=int, default=2000000,
                    help='Number of batches to run.')

parser.add_argument('--log_device_placement', type=bool, default=False,
                    help='Whether to log device placement.')

parser.add_argument('--log_frequency', type=int, default=10,
                    help='How often to log results to the console.')

def train():
    """Train CIFAR-10 for a number of steps."""
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False, name= 'global_step')

        # Get images and labels for CIFAR-10.
        images, labels, names = drd.distorted_inputs()

        # # Build an initialization operation to run below.
        init = tf.global_variables_initializer()
        # Start running operations on the Graph.
        sess = tf.Session(config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement))
        sess.run(init)

        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)
        names_list = []
        start_time = time.time()
        for i in range(0,20000):
            if i % 1000 == 0:
                print(i)
            names_im = sess.run([names])
            names_list.append(names_im)

        names = np.vstack(names_list).flatten()

        #print(names)
        print(type(names))
        print(names.shape)
        print(np.unique(names, return_counts=True))

        uniq, counts = np.unique(names, return_counts=True)

        print(uniq.shape)
        print(counts.shape)

        #print(names.flatten())
        #plt.imshow(im[0], interpolation='nearest')
        #plt.figure(figsize=(40, 40))
        #plt.show()
        #print(im)

def main(argv=None):  # pylint: disable=unused-argument
    train()


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    tf.app.run()
