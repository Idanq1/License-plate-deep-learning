from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import label_map_util, config_util
from object_detection.builders import model_builder
import tensorflow as tf
import numpy as np
import cv2

configs = config_util.get_configs_from_pipeline_file(r"ssd_mobilenet\pipeline.config")
detection_model = model_builder.build(model_config=configs["model"], is_training=False)

ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore("model\\ckpt-1").expect_partial()


@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections


category_index = label_map_util.create_category_index_from_labelmap(r"Dataset\label_map.pbtxt")
image = cv2.imread(r"Dataset\test\Images\testy.jpg")
image_np = np.array(image)
input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
detections = detect_fn(input_tensor)

num_detections = int(detections.pop('num_detections'))
detections = {key: value[0, :num_detections].numpy()
              for key, value in detections.items()}
detections['num_detections'] = num_detections

# detection_classes should be ints.
detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

label_id_offset = 1
image_np_with_detections = image_np.copy()

viz_utils.visualize_boxes_and_labels_on_image_array(
    image_np_with_detections,
    detections['detection_boxes'],
    detections['detection_classes'] + label_id_offset,
    detections['detection_scores'],
    category_index,
    use_normalized_coordinates=True,
    max_boxes_to_draw=5,
    min_score_thresh=.5,
    agnostic_mode=False)

cv2.imshow('object detection', image_np_with_detections)
cv2.waitKey(0)