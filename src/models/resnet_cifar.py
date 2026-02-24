import torch
import torch.nn as nn
import torchvision.models as models


class ResNetCIFAR(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.model = models.resnet18(weights=None)

        # CIFAR-10 images are 32x32, so modify first layer
        self.model.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.model.maxpool = nn.Identity()

        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)