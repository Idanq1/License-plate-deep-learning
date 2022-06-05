from tensorflow import keras
import tensorflow as tf
import numpy as np
import cv2
import os


def format_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    cv2.imshow("A", img)
    cv2.waitKey(0)
    img = cv2.resize(img, (50, 50))
    img = np.expand_dims(img, axis=0)
    return np.array(img)


def main():
    model = keras.models.load_model(r'..\Models\model3')
    test_folder = r"..\Dataset\tests"
    images_p = os.listdir(test_folder)
    for image in images_p:
        img = format_image(f"{test_folder}\\{image}")
        prediction = model.predict(img)[0]
        predict_prcnt = max(prediction)
        predict_digit = np.argmax(prediction)
        print(predict_digit)
        print(predict_prcnt)


if __name__ == '__main__':
    main()
