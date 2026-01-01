import torch
import torch.nn as nn
from torchvision import models


class ResNet18ROP(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        self.backbone = models.resnet18(pretrained=pretrained)
        in_features = self.backbone.fc.in_features

        # Replace final layer for binary classification
        self.backbone.fc = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.backbone(x)  # logits (no sigmoid)
