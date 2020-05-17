import numpy as np
import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.layer_fc1 = Dense(1280, 1024)
        self.layer_fc2 = Dense(1024, 256)
        self.layer_fc3 = Dense(256, 64)
        self.layer_output = nn.Linear(64, 1)
        self.activation_output = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_fc1(x)
        x = self.layer_fc2(x)
        x = self.layer_fc3(x)
        x = self.layer_output(x)
        x = self.activation_output(x)
        return x


class Dense(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(Dense, self).__init__()
        self.layer = nn.Linear(in_channels, out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer(x)
        x = self.activation(x)
        return x


if __name__ == '__main__':
    model = SimpleMLP()
    input_tensor: torch.Tensor = torch.rand(*(1, 1280), dtype=torch.float32, requires_grad=False)
    with torch.no_grad():
        output_tensor: torch.Tensor = model(input_tensor)
    output_array: np.ndarray = output_tensor.numpy()
    print(output_array)
