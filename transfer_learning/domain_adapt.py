import os
import tensorflow as tf
import keras
from keras import layers
import keras_cv
from keras.applications import EfficientNetB0
import datetime
import numpy as np

import matplotlib.pyplot as plt

print(f"keras version: {keras.__version__}")
print(f"keras_cv version: {keras_cv.__version__}")


# Constants
DATASET_NAME = "Modern-Office-31"
DATASET_DIR = f"/Users/mghifary/Work/Code/AI/data/{DATASET_NAME}"
BATCH_SIZE = 32
IMG_HEIGHT = 224
IMG_WIDTH = 224
SOURCE = "amazon"
TARGET = "webcam"

def visualize_images(ds, num_images=9):
    class_names = ds.class_names
    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(num_images):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")

# Load dataset
# Train
train_datadir = os.path.join(DATASET_DIR, SOURCE)
train_ds = keras.utils.image_dataset_from_directory(
    train_datadir,
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

# Data preprocessing
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

def input_preprocess_train(image, label):
    image = img_augmentation(image)
    label = tf.one_hot(label, num_classes)
    return image, label

def input_preprocess_test(image, label):
    label = tf.one_hot(label, num_classes)
    return image, label

train_ds = train_ds.map(input_preprocess_train, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

test_ds = test_ds.map(input_preprocess_test, num_parallel_calls=tf.data.AUTOTUNE)

# x_train = [x for x, _ in train_ds]
# y_train = [y for _, y in train_ds]
# x_train = np.concatenate(x_train, axis=0)
# y_train = np.concatenate(y_train, axis=0)

x_test = [x for x, _ in test_ds]
y_test = [y for _, y in test_ds]
x_test = np.concatenate(x_test, axis=0)
y_test = np.concatenate(y_test, axis=0)


# Build model
print(f"## Build model ##")

def build_efficientnet(num_classes, img_height, img_width, is_freezed=True):
    inputs = layers.Input(shape=(img_height, img_width, 3))
    backbone_model = EfficientNetB0(include_top=False, input_tensor=inputs, weights="imagenet")

    if is_freezed:
        # Freeze the pretrained weights
        backbone_model.trainable = False

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(backbone_model.output)
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="pred")(x)

    # Compile
    model = keras.Model(inputs, outputs, name="EfficientNet")
    optimizer = keras.optimizers.Adam(learning_rate=3e-4)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model

# Train the model
model = build_efficientnet(num_classes, IMG_HEIGHT, IMG_WIDTH, is_freezed=True)
print(model.summary())

# Define callback for Tensorboard and Model Checkpoint
MODELDIR = "models"
MPREFIX = f"{DATASET_NAME}_efficientnetB0-freezed_{SOURCE}-{TARGET}"
# MPREFIX = f"{DATASET_NAME}_resnet50-freezed_{SOURCE}-{TARGET}"
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard_callback = keras.callbacks.TensorBoard(
    log_dir=f"./logs/{MPREFIX}/{current_time}",
    update_freq="epoch"
)

checkpoint_dir = os.path.join(MODELDIR, f"{MPREFIX}-{current_time}")

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# checkpoint_filepath = os.path.join(checkpoint_dir, "checkpoint.weights.h5")
    checkpoint_filepath = os.path.join(checkpoint_dir, "model.keras")
modelcp_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
)

# # Store model config and architecture
# model_json = model.to_json()
# json_path = os.path.join(checkpoint_dir, "model.json")
# with open(json_path, "w") as json_file:
#     json_file.write(model_json)

epochs = 25
hist = model.fit(
    train_ds, 
    epochs=epochs, 
    validation_data=test_ds, 
    callbacks=[tensorboard_callback, modelcp_callback]
)