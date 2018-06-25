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

"""Routine for decoding the CIFAR-10 binary file format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import argparse
import tensorflow as tf

# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE = 448

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 5
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 2000 # was set from # 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 35

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='/home/olle/PycharmProjects/Diabetic_Retinopathy_Detection/data/balanced_512',
                    help='dir to train data')


FLAGS = parser.parse_args()


def read_svhn(filename_queue):
    """Reads and parses examples from SVHN data files.

    Recommendation: if you want N-way read parallelism, call this function
    N times.  This will give you N independent Readers reading different
    files & positions within those files, which will give better mixing of
    examples.

    Args:
      filename_queue: A queue of strings with the filenames to read from.

    Returns:
      An object representing a single example, with the following fields:
        height: number of rows in the result (32)
        width: number of columns in the result (32)
        depth: number of color channels in the result (3)
        key: a scalar string Tensor describing the filename & record number
          for this example.
        label: an int32 Tensor with the label in the range 0..9.
        uint8image: a [height, width, depth] uint8 Tensor with the image data
    """

    class SVHNRecord(object):
        pass

    result = SVHNRecord()

    # Dimensions of the images in the SVHN dataset.
    # See http://ufldl.stanford.edu/housenumbers/ for a description of the
    # input format.
    result.height = 512
    result.width = 512
    result.depth = 3

    reader = tf.TFRecordReader()
    result.key, value = reader.read(filename_queue)
    value = tf.parse_single_example(
        value,
        # Defaults are not specified since both keys are required.
        features={
            'image_raw': tf.FixedLenFeature(shape=[], dtype=tf.string),
            'label': tf.FixedLenFeature(shape=[], dtype=tf.int64),
            'image_name': tf.FixedLenFeature(shape=[], dtype=tf.string)
        })

    # Convert from a string to a vector of uint8 that is record_bytes long.
    record_bytes = tf.decode_raw(value['image_raw'], tf.uint8)
    # record_bytes.set_shape([32*32*3])
    record_bytes = tf.reshape(record_bytes, [result.height, result.width, 3])
    print("record bytes::::: ", record_bytes)
    # Store our label to result.label and convert to int32
    result.label = tf.cast(value['label'], tf.int32)
    result.name = tf.cast(value['image_name'], tf.string)
    result.uint8image = record_bytes

    return result


def _generate_image_and_label_batch(image, label,name, min_queue_examples,
                                    batch_size, shuffle):
  """Construct a queued batch of images and labels.

  Args:
    image: 3-D Tensor of [height, width, 3] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.

  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.
  num_preprocess_threads = 32
  if shuffle:
    images, label_batch, name_batch = tf.train.shuffle_batch(
        [image, label, name],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    images, label_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)

  # Display the training images in the visualizer.
  tf.summary.image('images', images)

  return images, tf.reshape(label_batch, [batch_size]), name_batch


def distorted_inputs(data_dir, batch_size):
    """Construct distorted input for SVHN training using the Reader ops.

    Args:
      data_dir: Path to the SVHN data directory.
      batch_size: Number of images per batch.

    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.
    """
    filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
                  for i in xrange(0, 10)]

    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    # #sppecifying angles for images to be rotated by
    # number_of_samples =

    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(filenames)
    # Read examples from files in the filename queue.
    read_input = read_svhn(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    # Image processing for training the network. Note the many random
    # distortions applied to the image.
    #crop out black pixels
    distorted_image = tf.image.central_crop(reshaped_image, 0.9)

    # Randomly crop a [height, width] section of the image.
    distorted_image = tf.random_crop(distorted_image, [height, width, 3])

    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    angles = tf.random_uniform([1], 0, 360, dtype=tf.float32, seed=0)

    distorted_image = tf.contrib.image.rotate(distorted_image, angles, interpolation='NEAREST', name=None)

    # Because these operations are not commutative, consider randomizing
    # the order their operation.
    # NOTE: since per_image_standardization zeros the mean and makes
    # the stddev unit, this likely has no effect see tensorflow#1458.

    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.divide(distorted_image, 255)

    #float_image = tf.image.adjust_brightness(float_image,0.25)
    # Set the shapes of tensors.
    float_image.set_shape([height, width, 3])
    # read_input.label.set_shape([1])58



    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)
    print('Filling queue with %d SVHN images before starting to train. '
          'This will take a few minutes.' % min_queue_examples)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(float_image, read_input.label, read_input.name,
                                           min_queue_examples, batch_size,
                                           shuffle=True)


def inputs(eval_data, batch_size):
    """Construct input for SVHN evaluation using the Reader ops.

    Args:
      eval_data: bool, indicating if one should use the train or eval data set.
      data_dir: Path
       to the SVHN data directory.
      batch_size: Number of images per batch.

    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.
    """
    data_dir = FLAGS.data_dir
    if not eval_data:
        filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
                     for i in xrange(0, 7)]
        for f in filenames:
            if not tf.gfile.Exists(f):
                raise ValueError('Failed to find file: ' + f)
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    else:
        # filenames = [os.path.join('/home/niyamat/Desktop/ml/data_to_upload_svhn', 'svhn_test.tfrecords')]
        filenames = [os.path.join('/home/olle/PycharmProjects/Diabetic_Retinopathy_Detection/data/validation', 'data_validation_batch.bin')]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(filenames)

    # Read examples from files in the filename queue.
    read_input = read_svhn(filename_queue)
    #reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    # Image processing for evaluation.
    # Crop the central [height, width] of the image.
    resized_image = tf.image.resize_image_with_crop_or_pad(read_input.uint8image,
                                                           height, width)

    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.divide(resized_image, 255)

    # Set the shapes of tensors.
    float_image.set_shape([height, width, 3])
    # read_input.label.set_shape([1])

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(float_image, read_input.label,
                                           min_queue_examples, batch_size,
                                           shuffle=False)