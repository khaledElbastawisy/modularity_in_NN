import tensorflow as tf 
import numpy as np 

def get_preprocessed_data(dataset):
    if dataset == 'cifar10':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        # Normalize pixel values to be between 0 and 1
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        # Convert labels to one-hot encoding
        y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

    if dataset == 'mnist':
        (x_train, y_train), (x_test, y_test)= tf.keras.datasets.mnist.load_data(dataset)
        x_train=x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
        x_train=x_train / 255.0
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
        x_test=x_test/255.0
        y_train = tf.one_hot(y_train.astype(np.int32), depth=10)
        y_test = tf.one_hot(y_test.astype(np.int32), depth=10)
        y_train= y_train.numpy()
        y_test= y_test.numpy()
    
    return x_train, y_train, x_test, y_test