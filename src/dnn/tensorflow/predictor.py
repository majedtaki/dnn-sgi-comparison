import numpy as np
import tensorflow as tf


class Predictor:
    def __init__(self, input_size: tuple = (1, 224, 224, 3)):
        self._loaded = tf.saved_model.load('./data/resnet50_tf/1')
        self._infer = self._loaded.signatures['serving_default']
        self._input_size = input_size
        self._output_name = 'probs'

    def predict(self) -> dict:
        input_data = np.random.rand(*self._input_size).astype(np.float32)
        output: tf.Tensor = self._infer(tf.constant(input_data))[self._output_name]
        return {
            "prediction": np.argmax(output.numpy(), 1).item()
        }
