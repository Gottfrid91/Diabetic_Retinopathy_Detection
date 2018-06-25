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

from datetime import datetime
import os.path
import time
import unicodedata
from matplotlib import pyplot as plt
from PIL import Image

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import drd
import sys
#write no pyc files
sys.dont_write_bytecode = True

parser = drd.parser

parser.add_argument('--train_dir', type=str, default='./output/resnet_2BLock_drd_train',
                    help='Directory where to write event logs and checkpoint.')

parser.add_argument('--pre_trained_dir', type=str, default='./output/pre_weights',
                    help='Directory where to write event logs and checkpoint.')

parser.add_argument('--max_steps', type=int, default=2000000,
                    help='Number of batches to run.')

parser.add_argument('--log_device_placement', type=bool, default=False,
                    help='Whether to log device placement.')

parser.add_argument('--log_frequency', type=int, default=10,
                    help='How often to log results to the console.')

parser.add_argument('--n_residual_blocks', type=int, default=5,
                    help='Number of residual blocks in network')


def train():
    """Train CIFAR-10 for a number of steps."""
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False, name= 'global_step')

        # Get images and labels for CIFAR-10.
        images, labels = drd.distorted_inputs()

        # Build a Graph that computes the logits predictions from the
        # inference model.
        #logits1= drd.inference(images, FLAGS.n_residual_blocks)
        logits = drd.oxford_net_C(images)

        # calculate predictions
        predictions = tf.cast(tf.argmax(logits, axis=1), tf.int32)

        # ops for batch accuracy calcultion
        correct_prediction = tf.equal(predictions, labels)
        batch_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # calculate training accuracy
        # Calculate loss.
        loss = drd.loss(logits, labels)

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_op = drd.train(loss, global_step)

        # Create a saver.
        saver = tf.train.Saver(tf.global_variables())

 #       sub_network = 'resnet_v1_2Blocks'
#        saver_30 = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=sub_network))
        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        # # Build an initialization operation to run below.
        init = tf.global_variables_initializer()
        # Start running operations on the Graph.
        sess = tf.Session(config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement))
        # sess.run(init)

        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

        step_start = 0
        try:
            ####Trying to find last checkpoint file fore full final model exist###
            print("Trying to restore last checkpoint ...")
            save_dir = FLAGS.train_dir
            # Use TensorFlow to find the latest checkpoint - if any.
            last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=save_dir)
            # Try and load the data in the checkpoint.
            saver.restore(sess, save_path=last_chk_path)

            # If we get to this point, the checkpoint was successfully loaded.
            print("Restored checkpoint from:", last_chk_path)
            # get the step integer from restored path to start step from there

            step_start = int(filter(str.isdigit, unicodedata.normalize('NFKD', last_chk_path).encode('ascii', 'ignore')))
            step_start = step_start #-20000
            print(step_start)

        except:
            try:
                #if first time model is being trained, load in pre-trained weights to subnetwork named pre_retina_30###
                print("Trying to load pretrained weights for "+sub_network+" network, init other random")
                pre_trained_dir = FLAGS.train_dir
                # Use TensorFlow to find the latest checkpoint - if any.
                pre_weights = tf.train.latest_checkpoint(checkpoint_dir=pre_trained_dir)
                #print(pre_weights)
                # Try and load the data in the checkpoint.
                saver_30.restore(sess, save_path=pre_weights)
                # If we get to this point, the checkpoint was successfully loaded.
                print("Restored checkpoint from:", pre_weights)
                #list all still unititlized variables

                uninitialized_vars = []
                for var in tf.global_variables():
                    try:
                        sess.run(var)
                    except tf.errors.FailedPreconditionError:
                        uninitialized_vars.append(var)

                #create init op for the still unitilized variables
                init_new_vars_op = tf.variables_initializer(uninitialized_vars)
                sess.run(init_new_vars_op)

                step_start = int(
                    filter(str.isdigit, unicodedata.normalize('NFKD', pre_weights).encode('ascii', 'ignore')))
            except:
                # If all the above failed for some reason, simply
                # initialize all the variables for the TensorFlow graph.
                print("Failed to restore any checkpoints. Initializing variables instead.")
                sess.run(init)

        accuracy_dev = []
        for step in xrange(step_start, FLAGS.max_steps):
            start_time = time.time()
            _, loss_value, accuracy, im = sess.run([train_op, loss, batch_accuracy, images])

            #append the next accuray to the development list
            accuracy_dev.append(accuracy)
            duration = time.time() - start_time
            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 10 == 0:
                num_examples_per_step = FLAGS.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d, loss = %.2f, avg_batch_accuracy = %.2f, (%.1f examples/sec; %.3f '
                              'sec/batch)')
                # take averages of all the accuracies from the previous bathces
                print(format_str % (datetime.now(), step, loss_value, np.mean(accuracy_dev),
                                    examples_per_sec, sec_per_batch))

            if step % 1000 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                #set paths and saving ops for the full and sub_network
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                pre_trained_path = os.path.join(FLAGS.pre_trained_dir, 'model.ckpt')

                saver.save(sess, checkpoint_path, global_step=step)
                #saver_30.save(sess, pre_trained_path, global_step=step)


def main(argv=None):  # pylint: disable=unused-argument
    train()


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    tf.app.run()
