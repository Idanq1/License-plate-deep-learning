from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import label_map_util
import tensorflow as tf
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
    # copy = img.copy()
    # print(img.shape)
    # og_height = img.shape[0]
    # og_width = img.shape[1]
    # og_size = (og_height, og_width)
    # ratio = og_height/og_width
    # new_size = 400
    # new_dim = (int(new_size/ratio), new_size)
    # img = cv2.resize(img, new_dim)
    # print("New", img.shape)
    return np.array(img)
    # return np.array(img), og_size, copy


images = os.listdir(r"..\Dataset\test\Images")
detect_fn = tf.saved_model.load(r"..\mobilenet_320_exported\saved_model")
image_path = r"..\Dataset\test\Images"
category_index = label_map_util.create_category_index_from_labelmap(r"..\Dataset\label_map.pbtxt", use_display_name=True)

s = time.time()

for image in images:
    print(f'Running inference for {image}... ')
    # image_np, original_size, og_image = load_image_into_numpy_array(f"{image_path}\\{image}")
    image_np = load_image_into_numpy_array(f"{image_path}\\{image}")
    width = image_np.shape[1]
    height = image_np.shape[0]
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
    # image_np_with_detections = og_image.copy()
    image_np_with_detections = image_np.copy()
    # viz_utils.visualize_boxes_and_labels_on_image_array(
    #       image_np_with_detections,
    #       detections['detection_boxes'],
    #       detections['detection_classes'],
    #       detections['detection_scores'],
    #       category_index,
    #       use_normalized_coordinates=True,
    #       max_boxes_to_draw=200,
    #       min_score_thresh=.30,
    #       agnostic_mode=False)
    i = 0
    # image_np_with_detections = cv2.resize(image_np_with_detections, tuple(reversed(original_size)))
    for box in detections["detection_boxes"]:
        score = detections["detection_scores"][i] * 100
        left, right, top, bottom = (int(box[1] * width), int(box[3] * width), int(box[0] * height), int(box[2] * height))
        i += 1
        if score > 30:
            pt1 = (right, top)
            pt2 = (left, bottom)
            image_np_with_detections = cv2.rectangle(image_np_with_detections, pt1, pt2, (182, 255, 62), 2)
            image_np_with_detections = cv2.putText(image_np_with_detections, f"{int(score)}%", (left, top), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 0.8, (15, 153, 255), 2)

    output_img = cv2.cvtColor(image_np_with_detections, cv2.COLOR_RGB2BGR)
    # cv2.imshow("output", output_img)
    cv2.imwrite(f"..\\Results\\{image}.png", output_img)
    # cv2.waitKey(0)
print(time.time() - s)
