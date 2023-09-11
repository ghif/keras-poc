import os
import numpy as np
import gzip

import tensorflow as tf
import keras

def load_data(data_dir=None):
    if data_dir is None:
        data_dir = "/Users/mghifary/Work/Code/AI/data/fashion_mnist"

    files = [
        "train-labels-idx1-ubyte.gz",
        "train-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
    ]

    # training label
    path = os.path.join(data_dir, files[0])
    with gzip.open(path, "rb") as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    path = os.path.join(data_dir, files[1])
    with gzip.open(path, "rb") as imgpath:
        x_train = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(
            len(y_train), 28, 28
        )

    path = os.path.join(data_dir, files[2])
    with gzip.open(path, "rb") as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    path = os.path.join(data_dir, files[3])
    with gzip.open(path, "rb") as imgpath:
        x_test = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(
            len(y_test), 28, 28
        )


    return (x_train, y_train), (x_test, y_test)

def preprocess_fashionmnist(ds, shuffle=False, augment=False, batch_size=32):
    AUTOTUNE = tf.data.AUTOTUNE

    rescale = keras.Sequential([
        keras.layers.Rescaling(1./255)
    ])

    data_augmentation = keras.Sequential([
        keras.layers.RandomZoom(0.1),
    ])

    # Rescale datasets
    ds = ds.map(
        lambda x, y: (rescale(x), y), 
        num_parallel_calls=AUTOTUNE
    )

    if shuffle:
        ds = ds.shuffle(buffer_size=1024)

    # Batch all datasets
    ds = ds.batch(batch_size)

    # Apply data augmentation
    if augment:
        ds = ds.map(
            lambda x, y: (data_augmentation(x, training=True), y),
            num_parallel_calls=AUTOTUNE
        )

    # Use buffered prefetching on all datasets
    return ds.prefetch(buffer_size=AUTOTUNE)