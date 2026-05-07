import torch.nn as nn
from torchvision import models


class VGG16Classifier(nn.Module):
    def __init__(self, num_classes=4):
        super(VGG16Classifier, self).__init__()

        # 不使用 pretrained weights，從零開始訓練
        self.model = models.vgg16(weights=None)

        # 修改最後一層分類器
        in_features = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)