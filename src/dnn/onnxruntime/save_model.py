import argparse
import torch
import torch.nn as nn
from torchvision.models.resnet import resnet50

from src.dnn.models import SimpleMLP


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet50')
    args = parser.parse_args()
    if args.model == 'resnet50':
        model: nn.Module = resnet50(pretrained=True)
        model_name = 'resnet50'
        x_dummy: torch.Tensor = torch.rand((1, 3, 224, 224))
    else:
        model: nn.Module = SimpleMLP()
        model_name = 'mlp'
        x_dummy: torch.Tensor = torch.rand((1, 1280))

    model.eval()
    torch.onnx.export(
        model,
        x_dummy,
        f'./data/{model_name}_pytorch.onnx',
        export_params=True,
        opset_version=10,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
    )


if __name__ == '__main__':
    main()
