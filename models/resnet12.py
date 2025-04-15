# models/resnet12.py
import torch.nn as nn
from torchvision.models.resnet import ResNet, BasicBlock

class ResNet12(ResNet):
    def __init__(self, pretrained=False, **kwargs):
        super().__init__(block=BasicBlock, layers=[1, 1, 1, 1], **kwargs)
        if pretrained:
            print("Warning: No pretrained weights for ResNet-12. Using random init.")
