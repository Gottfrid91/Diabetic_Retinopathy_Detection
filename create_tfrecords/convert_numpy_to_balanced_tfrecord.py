"""Converts MNIST data to TFRecords file format with Example protos."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import numpy as np
from PIL import ImageEnhance
import gc
import scipy as sio

import pandas as pd
from PIL import Image

import tensorflow as tf

save_directory = '/home/olle/PycharmProjects/Diabetic_Retinopathy_Detection/data/balanced'
label_directory = '/home/olle/PycharmProjects/Diabetic_Retinopathy_Detection/data/labels/'
data_directory = '/home/olle/PycharmProjects/Diabetic_Retinopathy_Detection/data/images/'
#@profile
def data_list(data_dir, label_dir, num_examples, k):
    '''
    imports: pandas, os, numpy, PIL
    '''
    class_factors = np.array([ 1, 12,  4,38, 47])

    width = 512
    height = 512
    # get labels csv into pandas df
    # below line assumes
    label_file_name = os.listdir(label_dir)[0]
    label_pd = pd.read_csv(label_dir + 'trainLabels.csv', engine='python')
    # initilize container list
    data = [[], [], [], [], []]
    # get filenames om images
    filenames = label_pd['image'].values
    print(type(filenames))
    # below loop retrieved the
    for im_number in range(k * num_examples, (k + 1) * num_examples):
        print(data_dir + filenames[im_number]+".jpeg")
        path = data_dir + filenames[im_number]+".jpeg"
        im = Image.open(path).resize((width, height), Image.ANTIALIAS)

        #preprocess images
        contrast = ImageEnhance.Contrast(im)
        img_contr = contrast.enhance(2)
        color = ImageEnhance.Color(img_contr)
        img_contr = color.enhance(0.4)
        brightness = ImageEnhance.Brightness(img_contr)
        img_bright = brightness.enhance(1)
        sharpness = ImageEnhance.Sharpness(img_bright)
        img_sharp = sharpness.enhance(2)
        #convert to numpy formatat before appending to list
        im = np.asarray(img_sharp).reshape(1, width, height, 3)
        name = filenames[im_number].replace(".jpeg", "")
        label = label_pd.loc[label_pd['image'] == name].iloc[0]['level']
        image_mean = np.mean(im)
        image_std = np.std(im)

        if label == 0:
            for i in range(class_factors[0]):
                data[0].append(name)
                data[1].append(im)
                data[2].append(label)
                data[3].append(image_mean)
                data[4].append(image_std)

        elif label == 1:
            for i in range(class_factors[1]):
                data[0].append(name)
                data[1].append(im)
                data[2].append(label)
                data[3].append(image_mean)
                data[4].append(image_std)

        elif label == 2:
            for i in range(class_factors[2]):
                data[0].append(name)
                data[1].append(im)
                data[2].append(label)
                data[3].append(image_mean)
                data[4].append(image_std)

        elif label == 3:
            for i in range(class_factors[3]):
                data[0].append(name)
                data[1].append(im)
                data[2].append(label)
                data[3].append(image_mean)
                data[4].append(image_std)

        elif label == 4:
            for i in range(class_factors[4]):
                data[0].append(name)
                data[1].append(im)
                data[2].append(label)
                data[3].append(image_mean)
                data[4].append(image_std)

        if im_number % 10 == 0:
            print("{} images loaded".format(im_number))
        gc.collect()

    return (data)

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _floats_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

#@profile
def convert_to(train_images,train_labels,train_name,train_num_exmaples,train_images_mean,train_images_std, name):
    """Converts a dataset to tfrecords."""
    images = train_images
    labels = train_labels
    print("train im shape is {}".format(images.shape))
    image_means = train_images_mean
    image_stds = train_images_std
    num_examples = train_num_exmaples
    print('number of examples this file is {}'.format(num_examples))
    if images.shape[0] != num_examples:
        raise ValueError('Images size %d does not match label size %d.' %
                         (images.shape[0], num_examples))
    rows = images.shape[1]
    cols = images.shape[2]
    depth = images.shape[3]

    filename = os.path.join(save_directory, name + '.bin')
    print('Writing', filename)
    with tf.python_io.TFRecordWriter(filename) as writer:
        for index in range(num_examples):
            if index%100 == 0:
                gc.collect()
            #print(images.shape)
            image_raw = images[index].tostring()
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'height': _int64_feature(rows),
                        'width': _int64_feature(cols),
                        'depth': _int64_feature(depth),
                        'image_mean': _floats_feature(image_means[index]),
                        'image_std': _floats_feature(image_stds[index]),
                        'label': _int64_feature(int(labels[index])),
                        'image_raw': _bytes_feature(image_raw),
                        'image_name': _bytes_feature(tf.compat.as_bytes(train_name[index])),

                    }))
            writer.write(example.SerializeToString())
#@profile
def main():
    filenames = os.listdir(data_directory)
    num_examples = int(len(filenames) / 10000)
    print("number of exmaples is {}".format(num_examples))
    for i in range(0, 10):

        data = data_list(data_directory, label_directory, num_examples, i)

        train_images = np.vstack(data[1])
        train_labels = np.asarray(data[2])
        train_name = data[0]
        train_num_exmaples = train_labels.shape[0]
        train_images_mean = np.asarray(data[3])
        train_images_std = np.asarray(data[4])
        print("convert new batch to tfrecords")
        # Convert to Examples and write the result to TFRecords.
        convert_to(train_images,train_labels,train_name,train_num_exmaples,train_images_mean,train_images_std, "data_batch_{}".format(i))
        del data
        del train_images
        gc.collect()

main()