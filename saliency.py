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

import time
from matplotlib import pyplot as plt

import numpy as np
import tensorflow as tf
import evaluation_functions as ef
import drd
import sys

sys.dont_write_bytecode = True

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', './output/_oxford_net_drd_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', './output/oxford_net_drd_train',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 4,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', True,
                         """Whether to run eval only once.""")

def saliency_map(output, input, name="saliency_map"):
    """
    Produce a saliency map as described in the paper:
    `Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps
    <https://arxiv.org/abs/1312.6034>`_.
    The saliency map is the gradient of the max element in output w.r.t input.
    Returns:
        tf.Tensor: the saliency map. Has the same shape as input.
    """
    max_outp = tf.reduce_max(output, 1)
    saliency_op = tf.abs(tf.gradients(max_outp, input)[:][0])
    return tf.identity(saliency_op, name=name)

def deprocess_image(X):
    return(np.multiply(X, 255))

def show_saliency_maps(X, y, pred,mask, saliency):
    mask = np.asarray(mask)
    Xm = X[mask]
    ym = y[mask]

    for i in range(mask.size):
        plt.subplot(2, mask.size, i + 1)
        plt.imshow(deprocess_image(Xm[i]))
        plt.axis('off')
        plt.title(ym[i])
        plt.subplot(2, mask.size, mask.size + i + 1)
        plt.title(pred[i])
        plt.imshow(saliency[i], cmap=plt.cm.hot)
        plt.axis('off')
        plt.gcf().set_size_inches(10, 4)
    plt.show()

def eval_once(saver,top_k_op, logits, images, labels, pred, saliency, target_conv,target_conv_layer_grad, gb_grad):
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
    images_list = []
    saliency_list = []
    t_conv_list = []
    t_conv_grad_list = []
    gb_gradient_list = []


    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))
      for i in range(0, FLAGS.num_examples):
        predictions, f_layer, cls_labels, cls_prediction, im, sal,t_conv, t_conv_grad, gb_gradient= sess.run([top_k_op,logits,
                                                                                     labels, pred,images,saliency,
                                                                                        target_conv,target_conv_layer_grad, gb_grad])
        #book keep prediction, logits, labels
        class_pred.append(cls_prediction)
        final_layer.append(f_layer)
        class_labels.append(cls_labels)
        images_list.append(im)
        saliency_list.append(sal)
        t_conv_list.append(t_conv)
        t_conv_grad_list.append(t_conv_grad)
        gb_gradient_list.append(gb_gradient)

      # convert bookkeeping to numpy for helper function
      class_labels = np.concatenate(class_labels, axis=0)
      final_layer = np.concatenate(final_layer, axis=0)
      class_pred = np.concatenate(class_pred, axis=0).reshape(FLAGS.num_examples, 1)
      images_list = np.concatenate(images_list, axis=0)
      saliency_list = np.concatenate(saliency_list, axis=0)
      t_conv_list = np.concatenate(t_conv_list, axis=0)
      t_conv_grad_list = np.concatenate(t_conv_grad_list, axis=0)
      gb_gradient_list = np.concatenate(gb_gradient_list, axis=0)

      mask = np.arange(FLAGS.num_examples)
      show_saliency_maps(images_list,class_labels,class_pred,mask,saliency_list)


      print(saliency_list.shape)
      print(t_conv_list.shape)
      print(images_list.shape)
      print(t_conv_grad_list.shape)
      print(gb_gradient_list.shape)
      #below loop shows the CAMgrad, guided backprop and guided grad_cam
      for i in range(FLAGS.num_examples):
          ef.visualize_gradients(images_list[i], t_conv_list[i], t_conv_grad_list[i], gb_gradient_list[i])

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
    logits, target_conv = drd.oxford_net_C(images)
    loss = drd.loss(logits, labels)
    # Calculate predictions.
    top_k_op = tf.nn.in_top_k(logits, labels, 1)
    pred = tf.cast(tf.argmax(logits, axis=1), tf.int32)
    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        drd.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)
    ########### Here the salienncy maps calculation begins############
    saliency = saliency_map(logits, images)

    y_c = tf.reduce_sum(tf.multiply(logits, tf.cast(labels, tf.float32)), axis=1)

    target_conv_layer_grad = tf.gradients(y_c, target_conv)[0]
    print('target_conv_layer_grad:', target_conv_layer_grad)
    # Guided backpropagtion back to input layer
    gb_grad = tf.gradients(loss, images)[0]
    print('gb_grad:', gb_grad)

    while True:
      eval_once(saver, top_k_op, logits, images, labels, pred, saliency,
                target_conv,target_conv_layer_grad, gb_grad)
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument
  evaluate()


if __name__ == '__main__':
  tf.app.run()
