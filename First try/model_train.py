from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.layers import Flatten, Dense, Input
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

images_path = r"dataset\images"
annots_path = r"dataset\annotations"

images = []
targets = []
filenames = []

annots_files = os.listdir(annots_path)
images_files = os.listdir(images_path)
print(images_files)
for file in images_files:
    base_name = os.path.splitext(file)[0]
    print(file)
    tree = ET.parse(f"{annots_path}\\{base_name}.xml")
    root = tree.getroot()
    try:
        x_start = root[4][5][0].text
        y_start = root[4][5][1].text
        x_end = root[4][5][2].text
        y_end = root[4][5][3].text
    except IndexError:
        x_start = root[6][4][0].text
        y_start = root[6][4][1].text
        x_end = root[6][4][2].text
        y_end = root[6][4][3].text
    print(f"{base_name}, {x_start}, {y_start}, {x_end}, {y_end}")

    img_path = f"{images_path}\\{file}"
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    x_start = float(x_start) / w
    y_start = float(y_start) / h
    x_end = float(x_end) / w
    y_end = float(y_end) / h

    img = load_img(img_path, target_size=(224, 224))
    img = img_to_array(img)

    images.append(img)
    targets.append((x_start, y_start, x_end, y_end))
    filenames.append(file)


images = np.array(images, dtype="float32") / 255.0
targets = np.array(targets, dtype="float32")

x_train, x_test, y_train, y_test, train_filenames, test_filenames = train_test_split(images, targets, filenames, test_size=0.1, random_state=42)
print(y_train)
print(y_test)

with open(r"test_filesnmames.txt", 'w') as f:
    f.write(("\n".join(test_filenames)))

vgg = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
vgg.trainable = False

flatten = vgg.output
flatten = Flatten()(flatten)

bboxHead = Dense(128, activation="relu")(flatten)
bboxHead = Dense(64, activation="relu")(bboxHead)
bboxHead = Dense(32, activation="relu")(bboxHead)
bboxHead = Dense(4, activation="sigmoid")(bboxHead)

model = Model(inputs=vgg.input, outputs=bboxHead)
opt = Adam(lr=1e-4,)
model.compile(loss="mse", optimizer=opt)
print(model.summary())

epochs = 10000
h = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, verbose=1)
model.save(r"model.h5", save_format="h5")

N = epochs
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), h.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), h.history["val_loss"], label="val_loss")
plt.title("Bounding Box Regression Loss on Training Set")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig(r"plot.png")
