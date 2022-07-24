class TfrSerializer:
    def __call__(self, raw_example):
        features = self.convert_to_feature(raw_example)
        features = tf.train.Features(feature=features)
        tf_example = tf.train.Example(features=features)
        serialized = tf_example.SerializeToString()
        return serialized

    def convert_to_feature(self, raw_example):
        pass

    def _bytes_feature(value):
        pass

    def _float_feature(value):
        pass

    def _int64_feature(value):
        pass
