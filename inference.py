import os
import numpy as np

import keras
import tensorflow as tf

def accuracy(y, y_pred):
    return np.mean(y == y_pred)


# Constants
BATCH = 128
DATADIR = "/Users/mghifary/Work/Code/AI/data"
MODELDIR = "/Users/mghifary/Work/Code/AI/models"
checkpoint_dir = os.path.join(MODELDIR, "mlp-mnist-20231222-180813")


# Load model from .keras
checkpoint_path = os.path.join(checkpoint_dir, "model.keras")
model = keras.models.load_model(checkpoint_path)

print(model.summary())

# Load dataset
data_path = os.path.join(DATADIR, "mnist.npz")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data(data_path)
x_train = x_train.astype("float32") / 255.
x_test = x_test.astype("float32") / 255.

# Reshape
x_train = np.expand_dims(x_train, axis=-1) 
x_test = np.expand_dims(x_test, axis=-1)

(n_train, dx1, dx2, c) = x_train.shape
n_test = x_test.shape[0]

ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))

# Shuffle and batch
ds_train = ds_train.shuffle(buffer_size=1024).batch(BATCH)
ds_test = ds_test.batch(BATCH)

# Predict and evaluate
print(f"Predict and evaluate model ...")
pred_train = model.predict(x_train)
y_pred_train = np.argmax(pred_train, axis=1).astype(np.uint8)

train_acc = accuracy(y_train, y_pred_train)
print(f" > Train accuracy: {train_acc:.4f}")

pred_test = model.predict(x_test)
y_pred_test = np.argmax(pred_test, axis=1).astype(np.uint8)

test_acc = accuracy(y_test, y_pred_test)
print(f" > Test accuracy: {test_acc:.4f}")


train_score = model.evaluate(x_train, y_train)
test_score = model.evaluate(x_test, y_test)
print(f"train_score: {train_score}")
print(f"test_score: {test_score}")