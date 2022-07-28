import tensorflow as tf
import numpy as np


class TfrSerializer:
    def __call__(self, raw_example):
        features = self.convert_to_feature(raw_example)
        features = tf.train.Features(feature=features)
        tf_example = tf.train.Example(features=features)
        serialized = tf_example.SerializeToString()
        return serialized

    def convert_to_feature(self, raw_example):
        features = dict()
        for key, value in raw_example.items():
            if value is None:
                continue
            elif isinstance(value, np.ndarray):
                value = value.tobytes()
                features[key] = self._bytes_feature(value)
            elif isinstance(value, str):
                value = bytes(value, 'utf-8')
                features[key] = self._bytes_feature(value)
            elif isinstance(value, int):
                features[key] = self._int64_feature(value)
            elif isinstance(value, float):
                features[key] = self._float_feature(value)
            else:
                assert 0, f"[convert_to_feature] Wrong data type: {type(value)}"
        return features

    @staticmethod
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


    @staticmethod
    def _float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    @staticmethod
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
