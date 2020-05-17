import numpy as np
import onnxruntime


class PredictorResNet50:
    def __init__(self, input_size: tuple = (1, 3, 224, 224)):
        self._ort_session = onnxruntime.InferenceSession('./data/resnet50_pytorch.onnx')
        self._input_size = input_size

    def predict(self) -> dict:
        input_data = np.random.rand(*self._input_size).astype(np.float32)
        output = self._ort_session.run(['output'], {'input': input_data})
        return {
            "prediction": np.argmax(output[0], axis=1).item()
        }


class PredictorMLP:
    def __init__(self, input_size: tuple = (1, 1280)):
        self._ort_session = onnxruntime.InferenceSession('./data/mlp_pytorch.onnx')
        self._input_size = input_size

    def predict(self) -> dict:
        input_data = np.random.rand(*self._input_size).astype(np.float32)
        output = self._ort_session.run(['output'], {'input': input_data})
        return {
            "prediction": output[0].item()
        }
