from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import label_map_util
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import numpy as np
import warnings
import time
import cv2
import os

warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings


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
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np.array(img)


images = os.listdir(r"Dataset\test\Images")
detect_fn = tf.saved_model.load(r"output2\saved_model")
image_path = r"Dataset\test\Images"
category_index = label_map_util.create_category_index_from_labelmap(r"Dataset\label_map.pbtxt", use_display_name=True)

s = time.time()

for image in images:
    print(f'Running inference for {image}... ')
    image_np = load_image_into_numpy_array(f"{image_path}\\{image}")

    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]
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
    # plt.figure()
    # plt.imshow(image_np_with_detections)
    # print('Done')
    # plt.show()
    # plt.savefig(f"Results\\{image}")
    output_img = cv2.cvtColor(image_np_with_detections, cv2.COLOR_RGB2BGR)
    cv2.imshow("output", output_img)
    cv2.waitKey(0)
print(time.time() - s)
# sphinx_gallery_thumbnail_number = 2