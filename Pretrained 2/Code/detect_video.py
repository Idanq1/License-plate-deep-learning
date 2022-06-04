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


def detect(image_np):
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
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

    return image_np_with_detections


def main():
    video = r"..\Dataset\Video\Raanana.webm"
    video = r"..\Dataset\Video\00091078-59817bb0.mov"
    output_path = r"..\Results\Video\output1.avi"
    cap = cv2.VideoCapture(video)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20, (frame_width, frame_height))
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"{i}\\{frames}")
            i += 1
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detected_image = detect(np.array(frame))
            output_img = cv2.cvtColor(detected_image, cv2.COLOR_RGB2BGR)

            out.write(output_img)
        else:
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    detect_fn = tf.saved_model.load(r"..\output2\saved_model")
    category_index = label_map_util.create_category_index_from_labelmap(r"..\Dataset\label_map.pbtxt", use_display_name=True)
    main()
