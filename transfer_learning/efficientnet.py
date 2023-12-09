from keras.applications import EfficientNetB0
import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
from keras import layers

import numpy as np
import datetime
import os

# Constants
IMG_SIZE = 224
BATCH_SIZE = 64
dataset_name = "stanford_dogs"


# # Load pretrained model
# model = EfficientNetB0(include_top=False, weights='imagenet')
# print(model.summary())

# Load dataset
(ds_train, ds_test), ds_info = tfds.load(
    dataset_name, split=["train", "test"], with_info=True, as_supervised=True
)
label_info = ds_info.features["label"]
num_classes = label_info.num_classes

# Resize images
size = (IMG_SIZE, IMG_SIZE)
ds_train = ds_train.map(lambda image, label: (tf.image.resize(image, size), label))
ds_test = ds_test.map(lambda image, label: (tf.image.resize(image, size), label))

# Visualizing the data
def format_label(label, label_info):
    string_label = label_info.int2str(label)
    return string_label.split("-")[1]

for i, (image, label) in enumerate(ds_train.take(9)):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image.numpy().astype("uint8"))
    plt.title("{}".format(format_label(label, label_info)))
    plt.axis("off")

# plt.show()

# Data augmentation
img_augmentation_Layers = [
    layers.RandomRotation(factor=0.15),
    layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
    layers.RandomFlip(),
    layers.RandomContrast(factor=0.1),
]

def img_augmentation(images):
    for layer in img_augmentation_Layers:
        images = layer(images)
    return images

# Show augmented image
for image, label in ds_train.take(1):
    for i in range(16):
        ax = plt.subplot(4, 4, i + 1)
        aug_img = img_augmentation(np.expand_dims(image.numpy(), axis=0))
        aug_img = np.array(aug_img)
        plt.imshow(aug_img[0].astype("uint8"))
        plt.title("{}".format(format_label(label, label_info)))
        plt.axis("off")

# plt.show()

# Preprocess inputs
def input_preprocess_train(image, label):
    image = img_augmentation(image)
    label = tf.one_hot(label, num_classes)
    return image, label

def input_preprocess_test(image, label):
    label = tf.one_hot(label, num_classes)
    return image, label

ds_train = ds_train.map(input_preprocess_train, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.batch(batch_size=BATCH_SIZE, drop_remainder=True)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

ds_test = ds_test.map(input_preprocess_test, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(batch_size=BATCH_SIZE, drop_remainder=True)

# Build deep model
def build_model(num_classes):
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    model = EfficientNetB0(include_top=False, input_tensor=inputs, weights="imagenet")

    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="pred")(x)

    # Compile
    model = keras.Model(inputs, outputs, name="EfficientNet")
    optimizer = keras.optimizers.Adam(learning_rate=1e-2)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model

# Train the model
model = build_model(num_classes=num_classes)

# Define callback for Tensorboard and Model Checkpoint
MODELDIR = "models"
MPREFIX = f"{dataset_name}_efficientnetB0-freezed"
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard_callback = keras.callbacks.TensorBoard(
    log_dir=f"./logs/{MPREFIX}/{current_time}",
    update_freq="epoch"
)


checkpoint_dir = os.path.join(MODELDIR, f"{MPREFIX}-{current_time}")

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

checkpoint_filepath = os.path.join(checkpoint_dir, "checkpoint.weights.h5")
modelcp_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True
)

epochs = 25
hist = model.fit(ds_train, epochs=epochs, validation_data=ds_test, callbacks=[tensorboard_callback, modelcp_callback])