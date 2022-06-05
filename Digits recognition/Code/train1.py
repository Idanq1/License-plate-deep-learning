import tensorflow.keras as keras
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
import tensorflow as tf
import numpy as np
import random
import time
import cv2
import os


def format_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (50, 50))
    return np.array(img)


def load_dataset(path):
    data = {}
    for label_f in os.listdir(path):
        if label_f not in data:
            data[label_f] = []
        for image in os.listdir(f"{path}\\{label_f}"):
            data[label_f].append(format_image(f"{path}\\{label_f}\\{image}"))

    images = []
    labels = []
    for label in data:
        for image in data[label]:
            images.append(image)
            labels.append(label)
    labels = np.array(labels)
    images = np.array(images) / 255.0
    images = images.reshape(*images.shape, 1)
    labels = to_categorical(labels, num_classes=10)
    return images, labels


def build_model():
    model = Sequential()
    model.add(Flatten(input_shape=(50, 50, 1)))
    model.add(Dense(128, activation="relu"))
    # model.add(BatchNormalization())
    model.add(Dense(128, activation="relu"))
    # model.add(BatchNormalization())
    # model.add(Dense(64, activation="relu"))
    model.add(Dense(10, activation="softmax"))

    # model.add(Activation('relu'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    # print(model.summary())

    return model


def main():
    s = time.time()

    batch_size = 64
    epochs = 15

    images, labels = load_dataset(r"..\Dataset\Digits")

    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.1, random_state=42, shuffle=True)

    model = build_model()
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, shuffle=True)

    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    model.save(r"..\Models\model3")

    print(time.time() - s)


if __name__ == '__main__':
    main()
