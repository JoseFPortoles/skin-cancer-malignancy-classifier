import torchvision.models as models
from torchvision.models import ResNet50_Weights, VGG16_Weights
import torch.nn as nn
from models.ext_unet import UNet


class SkinCancerClassifier(nn.Module):
    def __init__(self, resnet_weights=ResNet50_Weights.IMAGENET1K_V2, unet_weights=VGG16_Weights, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.unet = UNet(weights=unet_weights, out_channels=1)
        self.resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

    def forward(self, x):
        mask = self.unet(x).expand(3, -1, -1)
        x = x * mask
        return self.resnet(x)