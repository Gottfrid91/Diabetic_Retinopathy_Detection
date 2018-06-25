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

"""Builds the Diabetic_Retinopathy_Detection network.

Summary of available functions:

 # Compute input images and labels for training. If you would like to run
 # evaluations, use inputs() instead.
 inputs, labels = distorted_inputs()

 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)

 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)

 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import re

import tensorflow as tf
import drd_input
from resnet import *
import resnet_models as rm
import very_deep_oxford_net as ox_n
import sys

sys.dont_write_bytecode = True
parser = argparse.ArgumentParser()

# Basic model parameters.
parser.add_argument('--batch_size', type=int, default=1,
                    help='Number of images to process in a batch.')

parser.add_argument('--use_fp16', type=bool, default=False,
                    help='Train the model using fp16.')
# Basic model parameters.
parser.add_argument('--data_dir', type=str,
                    default='/home/olle/PycharmProjects/Diabetic_Retinopathy_Detection/data/validation',
                    help='Number of images to process in a batch.')

FLAGS = parser.parse_args()
# Global constants describing the digits-10 data set.
IMAGE_SIZE = drd_input.IMAGE_SIZE
NUM_CLASSES = drd_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = drd_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = drd_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.00001       # Initial learning rate.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity',
                                       tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))

  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def distorted_inputs():
    """Construct distorted input for SVHN training using the Reader ops.

    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.

    Raises:
      ValueError: If no data_dir
    """
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')

    images, labels, names = drd_input.distorted_inputs(data_dir=FLAGS.data_dir,batch_size=FLAGS.batch_size)
    if FLAGS.use_fp16:
        images = tf.cast(images, tf.float16)
        labels = tf.cast(labels, tf.float16)
    return images, labels, names

def inputs(eval_data):
    """Construct input for SVHN evaluation using the Reader ops.

    Args:
      eval_data: bool, indicating if one should use the train or eval data set.

    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.

    Raises:
      ValueError: If no data_dir
    """
    images, labels, names = drd_input.inputs(eval_data=eval_data,
                                       batch_size=FLAGS.batch_size, data_dir = FLAGS.data_dir)
    if FLAGS.use_fp16:
        images = tf.cast(images, tf.float16)
        labels = tf.cast(labels, tf.float16)
    return images, labels

def oxford_net_C(images):

    with tf.variable_scope('vgg_16'):
        layers = []
        with tf.variable_scope('conv1'):
            conv0 = ox_n.conv_bn_relu_layer(images, [3, 3, 3, 64], 1, scope_name='conv1_1')

            ox_n.activation_summary(conv0)
            layers.append(conv0)

            conv1 = ox_n.conv_bn_relu_layer(conv0, [3, 3, 64, 64], 1, scope_name='conv1_2')
            ox_n.activation_summary(conv1)
            layers.append(conv1)

            max_1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding= 'VALID', name='max_pool_1')
            layers.append(max_1)

        with tf.variable_scope('conv2'):
            conv2 = ox_n.conv_bn_relu_layer(max_1, [3, 3, 64, 128], 1, scope_name='conv2_1')
            ox_n.activation_summary(conv2)
            layers.append(conv2)

            conv3 = ox_n.conv_bn_relu_layer(conv2, [3, 3, 128, 128], 1, scope_name='conv2_2')
            ox_n.activation_summary(conv3)
            layers.append(conv3)

            max_2 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding= 'VALID', name='max_pool_2')

            layers.append(max_2)

        with tf.variable_scope('conv3'):
            conv3 = ox_n.conv_bn_relu_layer(max_2, [3, 3, 128, 256], 1, scope_name='conv3_1')
            ox_n.activation_summary(conv3)
            layers.append(conv3)

            conv4 = ox_n.conv_bn_relu_layer(conv3, [3, 3, 256, 256], 1, scope_name='conv3_2')
            ox_n.activation_summary(conv4)
            layers.append(conv4)

            conv5 = ox_n.conv_bn_relu_layer(conv4, [3, 3, 256, 256], 1, scope_name='conv3_3')
            ox_n.activation_summary(conv5)
            layers.append(conv5)

            max_3 = tf.nn.max_pool(conv5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='max_pool_3')

            layers.append(max_3)

        with tf.variable_scope('conv4'):
            conv6 = ox_n.conv_bn_relu_layer(max_3, [3, 3, 256, 512], 1, scope_name='conv4_1')
            ox_n.activation_summary(conv6)
            layers.append(conv6)

            conv7 = ox_n.conv_bn_relu_layer(conv6, [3, 3, 512, 512], 1, scope_name='conv4_2')
            ox_n.activation_summary(conv7)
            layers.append(conv7)

            conv8 = ox_n.conv_bn_relu_layer(conv7, [3, 3, 512, 512], 1, scope_name='conv4_3')
            ox_n.activation_summary(conv8)
            layers.append(conv8)

            max_4 = tf.nn.max_pool(conv8, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='max_pool_4')

            layers.append(max_4)

        with tf.variable_scope('conv5'):
            conv9 = ox_n.conv_bn_relu_layer(max_4, [3, 3, 512, 512], 1, scope_name='conv5_1')
            ox_n.activation_summary(conv9)
            layers.append(conv9)

            conv10 = ox_n.conv_bn_relu_layer(conv9, [3, 3, 512, 512], 1, scope_name='conv5_2')
            ox_n.activation_summary(conv10)
            layers.append(conv10)

            conv11 = ox_n.conv_bn_relu_layer(conv10, [3, 3, 512, 512], 1, scope_name='conv5_3')
            ox_n.activation_summary(conv11)
            layers.append(conv11)

            max_5 = tf.nn.max_pool(conv11, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='max_pool_5')

            layers.append(max_5)

        with tf.variable_scope('fc6'):
            in_channel = layers[-1].get_shape().as_list()[-1]
            bn_layer = ox_n.batch_normalization_layer(layers[-1], in_channel, scope_name='fc')
            relu_layer = tf.nn.relu(bn_layer)
            global_pool = tf.reduce_mean(relu_layer, [1, 2]) # avergage pooling

            fc1 = ox_n.output_layer(global_pool, 412, scope_name='fc1')
            layers.append(fc1)
        with tf.variable_scope('fc7'):
            fc2 = ox_n.output_layer(fc1, 312, scope_name='fc2')
            layers.append(fc2)
        with tf.variable_scope('fc8'):
            output = ox_n.output_layer(fc2, 5, scope_name='output')

            layers.append(output)

        return layers[-1]

def resnet_custom(input_tensor_batch, n, reuse=False):
    '''
    The main function that defines the ResNet. total layers = 1 + 2n + 2n + 2n +1 = 6n + 2
    :param input_tensor_batch: 4D tensor
    :param n: num_residual_blocks
    :param reuse: To build train graph, reuse=False. To build validation graph and share weights
    with train graph, resue=True
    :return: last layer in the network. Not softmax-ed
    '''
    with tf.variable_scope('resnet_custom'):

        layers = []
        with tf.variable_scope('conv0', reuse=reuse):
            conv0 = conv_bn_relu_layer(input_tensor_batch, [3, 3, 3, 16], 1)
            _activation_summary(conv0)
            layers.append(conv0)

        for i in range(n):
            with tf.variable_scope('conv1_%d' % i, reuse=reuse):
                if i == 0:
                    conv1 = residual_block(layers[-1], 16, first_block=True)
                else:
                    conv1 = residual_block(layers[-1], 16)
                _activation_summary(conv1)
                layers.append(conv1)

        for i in range(n):
            with tf.variable_scope('conv2_%d' % i, reuse=reuse):
                conv2 = residual_block(layers[-1], 32)
                _activation_summary(conv2)
                layers.append(conv2)

        for i in range(n):
            with tf.variable_scope('conv3_%d' % i, reuse=reuse):
                conv3 = residual_block(layers[-1], 64)
                layers.append(conv3)
#            assert conv3.get_shape().as_list()[1:] == [8, 8, 64]

    with tf.variable_scope('fc', reuse=reuse):
        in_channel = layers[-1].get_shape().as_list()[-1]
        bn_layer = batch_normalization_layer(layers[-1], in_channel)
        relu_layer = tf.nn.relu(bn_layer)
        global_pool = tf.reduce_mean(relu_layer, [1, 2])

        assert global_pool.get_shape().as_list()[-1:] == [64]
        output = output_layer(global_pool, 10)
        layers.append(output)

    return layers[-1]

def inference_2Blocks(images):
    return(rm.resnet_v1_2Blocks(images, num_classes=5, scope="resnet_v1_2Blocks"))

def resnet_v1_50(images):
    return(rm.resnet_v1_4Blocks(images, num_classes=5, scope="resnet_v1_50"))

def inference_alex_net(images):
    """Build the CIFAR-10 model.

    Args:
      images: Images returned from distorted_inputs() or inputs().

    Returns:
      Logits.
    """
    NUM_CLASSES = 5
    scope_model='alex_net'
    # We instantiate all variables using tf.get_variable() instead of
    # tf.Variable() in order to share variables across multiple GPU training runs.
    # If we only ran this model on a single GPU, we could simplify this function
    # by replacing all instances of tf.get_variable() with tf.Variable().
    #
    # conv1
    with tf.variable_scope(scope_model):
        with tf.variable_scope('conv1') as scope:
            kernel = _variable_with_weight_decay('weights',
                                                 shape=[5, 5, 3, 64],
                                                 stddev=5e-2,
                                                 wd=None)
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(pre_activation, name=scope.name)
            _activation_summary(conv1)

        # pool1
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool1')
        # norm1
        norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                          name='norm1')

        # conv2
        with tf.variable_scope('conv2') as scope:
            kernel = _variable_with_weight_decay('weights',
                                                 shape=[5, 5, 64, 64],
                                                 stddev=5e-2,
                                                 wd=None)
            conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv2 = tf.nn.relu(pre_activation, name=scope.name)
            _activation_summary(conv2)

        # norm2
        norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                          name='norm2')
        # pool2
        pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                               strides=[1, 2, 2, 1], padding='SAME', name='pool2')

        # local3
        with tf.variable_scope('local3') as scope:
            # Move everything into depth so we can perform a single matrix multiply.
            reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
            dim = reshape.get_shape()[1].value
            weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                                  stddev=0.04, wd=0.004)
            biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
            local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
            _activation_summary(local3)

        # local4
        with tf.variable_scope('local4') as scope:
            weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                                  stddev=0.04, wd=0.004)
            biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
            local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
            _activation_summary(local4)

        # linear layer(WX + b),
        # We don't apply softmax here because
        # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
        # and performs the softmax internally for efficiency.
        with tf.variable_scope('softmax_linear') as scope:
            weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES],
                                                  stddev=1 / 192.0, wd=None)
            biases = _variable_on_cpu('biases', [NUM_CLASSES],
                                      tf.constant_initializer(0.0))
            softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
            _activation_summary(softmax_linear)

    return softmax_linear

def loss(logits, labels):
  """Add L2Loss to all the trainable variables.

  Add summary for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]

  Returns:
    Loss tensor of type float.
  """

  #code below multiplies logits after there labels
  # weights_initializer = tf.constant_initializer(value=[1.28205128, 12.5, 11.11111111, 33.33333333, 50.])
  # class_weights = tf.get_variable(name="bias_one_tf_var",
  #                            shape=[5],
  #                            initializer=weights_initializer)  # Calculate the average cross entropy loss across the batch.
  # logits_weighted = tf.multiply(
  #     logits,
  #     class_weights,
  #     name="class_weighing"
  # )
  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)


  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
  """Add summaries for losses in digits-10 model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.summary.scalar(l.op.name + ' (raw)', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))

  return loss_averages_op


def train(total_loss, global_step):
  """Train digits-10 model.

  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.

  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # Variables that affect learning rate.
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  tf.summary.scalar('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    opt = tf.train.AdamOptimizer(lr)
    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.summary.histogram(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op
