import os
import keras
import matplotlib.pyplot as plt
import tensorflow as tf

# Constants
DATASET_DIR = "<directory to the dataset>"
BATCH_SIZE = 32
IMG_HEIGHT = 224
IMG_WIDTH = 224

# Load dataset
# Train
train_datadir = os.path.join(DATASET_DIR, "amazon")
train_ds = keras.utils.image_dataset_from_directory(
    train_datadir,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
)

# Test
test_datadir = os.path.join(DATASET_DIR, "dslr")
test_ds = keras.utils.image_dataset_from_directory(
    test_datadir, 
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

# Check class names between train (source) and test (target) domain - should be the same
for ctrain, ctest in zip(train_ds.class_names, test_ds.class_names):
    assert ctrain == ctest

print(f"Train and test classes are the same: {len(train_ds.class_names)} classes")


# Visualize the data
def visualize_images(ds, num_images=9):
    class_names = ds.class_names
    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(num_images):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")

visualize_images(train_ds)
plt.show()

visualize_images(test_ds)
plt.show()