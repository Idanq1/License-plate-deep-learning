from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
import os


def format_image(img_path):
    img = load_img(img_path, target_size=(224, 224))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def main():
    images_path = r"testing"
    model_path = r"model_1500.h5"
    model = load_model(model_path)
    for file in os.listdir(images_path):
        img_path = f"testing\\{file}"
        img = format_image(img_path)
        predict = model.predict(img)[0]
        st_x, st_y, end_x, end_y = predict
        image = cv2.imread(img_path)
        try:
            image = imutils.resize(image, width=600)
        except AttributeError:
            print(file)
            raise AttributeError
        h, w = image.shape[:2]

        st_x = int(st_x * w)
        st_y = int(st_y * h)
        end_x = int(end_x * w)
        end_y = int(end_y * h)
        cv2.rectangle(image, (st_x, st_y), (end_x, end_y), (0, 255, 0), 2)

        cv2.imshow("output", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

