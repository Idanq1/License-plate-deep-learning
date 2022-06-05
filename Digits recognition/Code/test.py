import tensorflow as tf  # deep learning library. Tensors are just multi-dimensional arrays
import numpy as np

mnist = tf.keras.datasets.mnist  # mnist is a dataset of 28x28 images of handwritten digits and their labels
(x_train, y_train),(x_test, y_test) = mnist.load_data()  # unpacks images to x_train/x_test and labels to y_train/y_test
print(type(x_train))
