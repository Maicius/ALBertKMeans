import os
import sys
import tensorflow as tf
pwd = os.path.dirname(os.path.abspath("../"))
sys.path.append(pwd)
from src.albert import modeling
from src.albert.hyperparameters import Hyperparamters as hp

bert_config_file = os.path.join(hp.bert_path, 'albert_config.json')
bert_config = modeling.AlbertConfig.from_json_file(bert_config_file)


class AlbertEncodeModel(object):
    def __init__(self, is_training):
        self.is_training = is_training

        self.input_ids = tf.placeholder(tf.int32, shape=[None, hp.seq_len], name='input_ids')
        self.input_masks = tf.placeholder(tf.int32, shape=[None, hp.seq_len], name='input_masks')
        self.input_segment_ids = tf.placeholder(tf.int32, shape=[None, hp.seq_len], name='input_segment_ids')

        self.model = modeling.AlbertModel(
            config=bert_config,
            is_training=self.is_training,
            input_ids=self.input_ids,
            input_mask=self.input_masks,
            token_type_ids=self.input_segment_ids,
            use_one_hot_embeddings=False)
        # 得到编码输出
        self.ouput_layer = self.model.get_pooled_output()

        tvars = tf.trainable_variables()
        # 初次加载模型
        if hp.init_checkpoint:
            (assignment_map, initialized_variable_names) = \
                modeling.get_assignment_map_from_checkpoint(tvars,
                                                            hp.init_checkpoint)
            tf.train.init_from_checkpoint(hp.init_checkpoint, assignment_map)


    def get_pooled_output(self):
        return self.ouput_layer
