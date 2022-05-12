import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format


def update_pipeline(config_path):
    # config = config_util.get_configs_from_pipeline_file(config_path)
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.io.gfile.GFile(config_path, 'r') as f:
        proto_str = f.read()
        text_format.Merge(proto_str, pipeline_config)
    pipeline_config.model.ssd.num_classes = 1
    pipeline_config.train_config.batch_size = 12
    pipeline_config.train_config.fine_tune_checkpoint = r'ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8\checkpoint\ckpt-0'
    pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
    pipeline_config.train_input_reader.label_map_path = r"Dataset\label_map.pbtxt"
    pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [r"Dataset\train.record"]
    pipeline_config.eval_input_reader[0].label_map_path = r"Dataset\label_map.pbtxt"
    pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [r"Dataset\test.record"]
    config_text = text_format.MessageToString(pipeline_config)
    with tf.io.gfile.GFile(config_path, "wb") as f:
        f.write(config_text)


update_pipeline(r"ssd_mobilenet\\pipeline.config")

# python model\models\research\object_detection\model_main_tf2.py --model_dir=ssd_mobilenet --pipeline_cofig_path=ssd_mobilenet\pipeline.config --num_train_steps=5000
