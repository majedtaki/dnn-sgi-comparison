import grpc
import numpy as np
from tensorflow_core import make_tensor_proto
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc


class ProxyTFServing:
    def __init__(self, host: str = 'localhost', port: int = 8500, input_size: tuple = (1, 224, 224, 3)):
        endpoint = host + ':' + str(port)
        channel = grpc.insecure_channel(endpoint)
        self._stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
        self._input_size = input_size

    def predict(self):
        request: predict_pb2.PredictRequest = self._prepare_request()
        response: predict_pb2.PredictResponse = self._stub.Predict(request, 30)
        output: np.ndarray = response.outputs['probs'].float_val
        return {
            'predictions': np.argmax(list(output), 0).item()
        }

    def _prepare_request(self) -> predict_pb2.PredictRequest:
        input_data = np.random.rand(*self._input_size[1:]).astype(np.float32)
        tensor = make_tensor_proto(input_data, shape=list(self._input_size))
        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'resnet50_tf'
        request.model_spec.signature_name = 'serving_default'
        request.inputs['input_1'].CopyFrom(tensor)
        return request
