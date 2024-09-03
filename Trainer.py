import numpy as np
import os
import sys
import tensorflow as tf
import pickle

from sklearn.model_selection import train_test_split

EPOCHS = 3
IMG_WIDTH = 28
IMG_HEIGHT = 28
NUM_CATEGORIES = 7
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    # if len(sys.argv) not in [2, 3]:
    #     sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data()

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    # if len(sys.argv) == 3:
    #     filename = sys.argv[2]
    model.save("model.h5")
        # print(f"Model saved to {filename}.")


def load_data():
    with open("features", "rb") as f:
        images = pickle.load(f)
    with open ("labels", "rb") as f:
        labels = pickle.load(f)

    return images, labels


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 1)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 1)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(256, activation="relu"),
        # tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.6),

        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    main()
