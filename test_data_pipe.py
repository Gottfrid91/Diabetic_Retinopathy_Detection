import tensorflow as tf
import drd_input as i
import matplotlib.pyplot as plt
import numpy as np
data_dir = "/home/olle/PycharmProjects/Diabetic_Retinopathy_Detection/data/balanced"

for example in tf.python_io.tf_record_iterator(data_dir+"/data_batch_3.bin"):
    result = tf.train.Example.FromString(example)
    print(result.features.feature['label'].int64_list.value)
    print(result.features.feature['image_name'].bytes_list.value)

filename_queue = tf.train.string_input_producer([data_dir+"/data_batch_3.bin"])
read_input = i.read_svhn(filename_queue)
sess = tf.InteractiveSession()
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord)
img, lab, id = sess.run([read_input.uint8image, read_input.label, read_input.name])
img = np.asarray(img).reshape(1, img.shape[0],img.shape[1], 3)
lab = np.asarray(lab)
print(img.shape,lab, id)#, label.shape, name)

# Loop over each example in batch
for i in range(img.shape[0]):
    plt.imshow(img[i])
    plt.show()
    #print('Class label ' + lab)