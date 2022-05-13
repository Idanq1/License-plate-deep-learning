import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import cv2
import warnings
warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings


# image_path = r"Dataset\test\Images\testy.jpg"
image_path = r"_PAZ5328JPG.jpg"


def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(Image.open(path))


images = os.listdir(r"Dataset\test\Images")
images = ["white.jpg"]

for image in images:
    image_path = r"Dataset\test\Images"
    detect_fn = tf.saved_model.load(r"output\saved_model")
    category_index = label_map_util.create_category_index_from_labelmap(r"Dataset\label_map.pbtxt", use_display_name=True)

    print('Running inference for {}... '.format(image), end='')
    image_np = load_image_into_numpy_array(f"{image_path}\\{image}")

    # Things to try:
    # Flip horizontally
    # image_np = np.fliplr(image_np).copy()
    # Convert image to grayscale
    # image_np = np.tile(
    #     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image_np)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]
    # input_tensor = np.expand_dims(image_np, 0)
    detections = detect_fn(input_tensor)
    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                   for key, value in detections.items()}
    detections['num_detections'] = num_detections
    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    image_np_with_detections = image_np.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          detections['detection_boxes'],
          detections['detection_classes'],
          detections['detection_scores'],
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=200,
          min_score_thresh=.30,
          agnostic_mode=False)
    plt.figure()
    plt.imshow(image_np_with_detections)
    print('Done')
    plt.show()
    plt.savefig(f"Results\\{image}")
    # cv2.imshow("output", image_np_with_detections)
    # cv2.waitKey(0)

# sphinx_gallery_thumbnail_number = 2