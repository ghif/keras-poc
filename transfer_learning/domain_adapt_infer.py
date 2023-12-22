import os
import numpy as np

import tensorflow as tf
import keras
from keras import layers
from keras.applications import EfficientNetB0

SOURCE = "amazon"
TARGET = "webcam"

IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 128

DATASET_NAME = "Modern-Office-31"
DATASET_DIR = f"/Users/mghifary/Work/Code/AI/data/{DATASET_NAME}"

MODELDIR = "models"
MPREFIX = f"{DATASET_NAME}_efficientnetB0-freezed_{SOURCE}-{TARGET}"

def input_preprocess(image, label):
    label = tf.one_hot(label, num_classes)
    return image, label

# Load dataset
# Train
train_datadir = os.path.join(DATASET_DIR, SOURCE)
train_ds = keras.utils.image_dataset_from_directory(
    train_datadir,
    shuffle=False,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
)


# Test
test_datadir = os.path.join(DATASET_DIR, TARGET)
test_ds = keras.utils.image_dataset_from_directory(
    test_datadir, 
    shuffle=False,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)


# Check class names between train (source) and test (target) domain - should be the same
for ctrain, ctest in zip(train_ds.class_names, test_ds.class_names):
    assert ctrain == ctest

num_classes = len(train_ds.class_names)
print(f"Train ({SOURCE}) and test ({TARGET}) classes are the same: {num_classes} classes")

# Preprocess inputs
train_ds = train_ds.map(input_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.map(input_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

# Load model architecture and weights
checkpoint_dir = os.path.join(MODELDIR, f"{MPREFIX}-20231222-215114")
checkpoint_path = os.path.join(checkpoint_dir, "model.keras")
model = keras.models.load_model(checkpoint_path)
print(model.summary())

# Predict label
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

train_score = model.evaluate(train_ds)
test_score = model.evaluate(test_ds)

print(train_score)
print(test_score)

# Data concatenation
x_train = [x for x, _ in train_ds]
y_train = [y for _, y in train_ds]
x_train = np.concatenate(x_train, axis=0)
y_train = np.concatenate(y_train, axis=0)

x_test = [x for x, _ in test_ds]
y_test = [y for _, y in test_ds]
x_test = np.concatenate(x_test, axis=0)
y_test = np.concatenate(y_test, axis=0)


train_score2 = model.evaluate(x=x_train, y=y_train)
test_score2 = model.evaluate(x=x_test, y=y_test)

print(train_score2)
print(test_score2)

print("## Predict label and evaluate model ##")

print(" -- on source domain")
y_train = np.argmax(y_train, axis=1)
train_pred = model.predict(x_train, batch_size=BATCH_SIZE)
y_train_pred = np.argmax(train_pred, axis=1)

train_acc = accuracy(y_train, y_train_pred)
print(f"Source accuracy: {train_acc}")

print(" -- on target domain")
y_test = np.argmax(y_test, axis=1)
test_pred = model.predict(x_test, batch_size=BATCH_SIZE)
y_test_pred = np.argmax(test_pred, axis=1)


test_acc = accuracy(y_test, y_test_pred)
print(f"Target accuracy: {test_acc}")