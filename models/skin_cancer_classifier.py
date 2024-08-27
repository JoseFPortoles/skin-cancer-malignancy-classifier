import torchvision.models as models
from torchvision.models import ResNet50_Weights, VGG16_Weights
import torch.nn as nn
from models.ext_unet import UNet
from models.helpers import init_weights


class SkinCancerClassifier(nn.Module):
    def __init__(self, resnet_weights=ResNet50_Weights.IMAGENET1K_V2, unet_weights=VGG16_Weights, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.unet = UNet(weights=unet_weights, out_channels=1)
        self.resnet = models.resnet50(weights=resnet_weights)
        self.resnet.fc = nn.Linear(2048, 2, True).apply(init_weights)

    def forward(self, x):
        mask = self.unet(x).expand(-1, 3, -1, -1)
        x = x * mask
        return nn.Softmax(dim=1)(self.resnet(x))