import torchvision.models as models
from torchvision.models import ResNet50_Weights

resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

