# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Evaluation for CIFAR-10.

Accuracy:
cifar10_train.py achieves 83.0% accuracy after 100K steps (256 epochs
of data) as judged by cifar10_eval.py.

Speed:
On a single Tesla K40, cifar10_train.py processes a single batch of 128 images
in 0.25-0.35 sec (i.e. 350 - 600 images /sec). The model reaches ~86%
accuracy after 100K steps in 8 hours of training time.

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf

import drd
import t_sne
import evaluation_functions as ef
import sys

sys.dont_write_bytecode = True

parser = drd.parser

parser.add_argument('--eval_dir', type=str, default = './output/_oxford_net_drd_eval',
                          help = """Directory where to write event logs.""")
parser.add_argument('--eval_data', type = str, default='test',
                           help = """Either 'test' or 'train_eval'.""")
parser.add_argument('--checkpoint_dir', type=str, default= './output/oxford_net_drd_train',
                          help= """Directory where to read model checkpoints.""")
parser.add_argument('--eval_interval_secs', type= int, default= 60,
                            help="""How often to run the eval.""")
parser.add_argument('--num_examples',type=int, default=1000,
                            help="""Number of examples to run.""")
parser.add_argument('--run_once', type=bool, default=True,
                    help='Whether to run eval only once.')
parser.add_argument('--n_residual_blocks', type=int, default=5,
                    help='Number of residual blocks in network')

def eval_once(saver, summary_writer, top_k_op, summary_op, logits, images, labels, pred):
  """Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op
  """
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      print(ckpt )
      print(ckpt.model_checkpoint_path)
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/cifar10_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return

    #lists to append results for visualizations
    final_layer = []
    class_pred = []
    class_labels = []
    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
      true_count = 0  # Counts the number of correct predictions.
      total_sample_count = num_iter * FLAGS.batch_size
      step = 0
      while step < num_iter and not coord.should_stop():
        if step % 100 == 0:
          print("The step is {}".format(step))
        predictions, f_layer, cls_labels, cls_prediction = sess.run([top_k_op,logits, labels, pred])
        #book keep prediction, logits, labels
        class_pred.append(cls_prediction)
        final_layer.append(f_layer)
        class_labels.append(cls_labels)


        true_count += np.sum(predictions)
        step += 1

      # Compute precision @ 1.
      precision = true_count / total_sample_count
      print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))
      #convert bookkeeping to numpy for helper function
      class_labels  = np.concatenate(class_labels, axis=0 )
      final_layer = np.concatenate(final_layer, axis=0)
      class_pred = np.concatenate(class_pred, axis=0 ).reshape(FLAGS.num_examples,1)
      print('the weighted kappa metric is {}'.format(ef.quadratic_kappa(class_labels, class_pred.reshape(FLAGS.num_examples))))
      #here insert the TSNET visualization
      t_sne.plot_embedding(t_sne.t_sne_fit(final_layer), sess.run(images),
                           class_labels,
                           iter, title= "Final layer representation")

      ef.plot_confusion_matrix(class_pred, class_labels, 5)

      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='Precision @ 1', simple_value=precision)
      summary_writer.add_summary(summary, global_step)
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def evaluate():
  """Eval CIFAR-10 for a number of steps."""
  with tf.Graph().as_default() as g:
    # Get images and labels for CIFAR-10.
    eval_data = FLAGS.eval_data == 'test'

    images, labels = drd.inputs(eval_data=eval_data)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    #logits = drd.inference(images, FLAGS.n_residual_blocks)
    logits = drd.oxford_net_C(images)
    # Calculate predictions.
    top_k_op = tf.nn.in_top_k(logits, labels, 1)
    pred = tf.cast(tf.argmax(logits, axis=1), tf.int32)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        drd.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

    while True:
      eval_once(saver, summary_writer, top_k_op, summary_op, logits, images, labels, pred)
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument
  evaluate()


if __name__ == '__main__':
  FLAGS = parser.parse_args()
  tf.app.run()
