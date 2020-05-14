import numpy as np
import torch
import torch.nn as nn
from torchvision.models.resnet import resnet50


class Predictor:
    def __init__(self, input_size: tuple = (1, 3, 224, 224)):
        self._model: nn.Module = resnet50(pretrained=True)
        self._model.eval()
        self._input_size = input_size

    def predict(self) -> dict:
        input_data = np.random.rand(*self._input_size).astype(np.float32)
        tensor: torch.Tensor = torch.from_numpy(input_data)
        output: torch.Tensor = self._model(tensor)
        _, pred = output.max(1)
        return {
            "prediction": pred.item()
        }
