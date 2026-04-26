import torch.nn as nn
from torchvision import models

def get_efficientnet_b0(num_classes=4, pretrained=True):
    if pretrained:
        weights = models.EfficientNet_B0_Weights.DEFAULT
    else:
        weights = None

    model = models.efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model