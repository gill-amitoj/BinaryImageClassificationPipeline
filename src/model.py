# ----------------------------------------------------------
# Intelligent Image Classification System
# Author: Amitoj Singh (CCID: amitoj3)
# ----------------------------------------------------------

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

def build_model(num_classes: int) -> nn.Module:
    """
    Build a ResNet-18 transfer learning model.

    - Loads ImageNet-pretrained weights using the modern torchvision API
    - Freezes the backbone
    - Replaces the final fully-connected head for `num_classes`
    """
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)

    # Freeze backbone
    for param in model.parameters():
        param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes)
    )
    return model
