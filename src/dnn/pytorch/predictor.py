import numpy as np
import torch
import torch.nn as nn
from torchvision.models.resnet import resnet50

from src.dnn.models import SimpleMLP


class PredictorResNet50:
    def __init__(self, input_size: tuple = (1, 3, 224, 224)):
        self._model: nn.Module = resnet50(pretrained=True)
        self._model.eval()
        self._input_size = input_size

    def predict(self) -> dict:
        with torch.no_grad():
            tensor: torch.Tensor = torch.rand(*self._input_size, dtype=torch.float32)
            output: torch.Tensor = self._model(tensor)
            pred: np.ndarray = output.numpy()
        return {
            "prediction": np.argmax(pred, 1).item()
        }


class PredictorMLP:
    def __init__(self, input_size: tuple = (1, 1280)):
        self._model = SimpleMLP()
        self._model.eval()
        self._input_size = input_size

    def predict(self) -> dict:
        with torch.no_grad():
            tensor: torch.Tensor = torch.rand(*self._input_size, dtype=torch.float32)
            output: torch.Tensor = self._model(tensor)
            pred: np.ndarray = output.numpy()
        return {
            "prediction": pred[0].item()
        }
