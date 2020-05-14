import torch
import torch.nn as nn
from torchvision.models.resnet import resnet50


def main():
    model: nn.Module = resnet50(pretrained=True)
    model.eval()
    x_dummy: torch.Tensor = torch.rand((1, 3, 224, 224))
    torch.onnx.export(
        model,
        x_dummy,
        './data/resnet50_pytorch.onnx',
        export_params=True,
        opset_version=10,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
    )


if __name__ == '__main__':
    main()
