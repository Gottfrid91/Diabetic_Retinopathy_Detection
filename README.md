# Diabetic_Retinopathy_Detection
This is a TensorFlow example implementation for retina classification of diabetes. The network is a standard residual network model with 50 layers which is defined in resnet_models.py and resnet_utils.py.

### Models
This repository contains 5 model configurations. One deep and one shallow Resnet config, vgg19 config and alexnet. The different archetectures can be trained in drd_train_resnet_v1_50.py, drd_train_resnet_v1_30.py, drd_train_oxford_net.py, drd_train_shallow_oxford_net.py, drd_train_alex_net.py.

### Create Data
given the data is downloaded, the TFrecords can be created from the scripts in folder: create_tfrecords. One scripts creates a balanced data set. 

### Other files
See also different experimental jupyter notebooks in notebooks. and papers work is based on in papers.
