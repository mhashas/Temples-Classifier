import torch
import torch.nn as nn
from torchvision import models

from constants import *

class ResNet(nn.Module):
    def __init__(self, args):
        super(ResNet, self).__init__()
        self.args = args

        if args.model == RESNET:
            resnet = models.resnet101(pretrained=args.pretrained)
        elif args.model == RESNET_50:
            resnet = models.resnet50(pretrained=args.pretrained)
        else:
            raise NotImplementedError()

        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.backbone = nn.Sequential(self.layer0, resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)

        return x

    def get_feature_maps_size(self):
        return 2048 * 1 * 1
