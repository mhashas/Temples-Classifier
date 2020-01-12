import torch
import torch.nn as nn
import torchvision
from constants import *

class VGG(nn.Module):


    def __init__(self, args, pretrained):
        super(VGG, self).__init__()
        self.args = args
        if args.model == VGG:
            vgg = torchvision.models.vgg16_bn(pretrained=pretrained)
        else:
            vgg = torchvision.models.vgg19_bn(pretrained=pretrained)

        self.model = vgg.features
        self.avgpool = nn.AdaptiveAvgPool2d((4,4))


    def forward(self, x):
        x = self.model(x)
        x = self.avgpool(x)

        return x

    def get_feature_maps_size(self):
        return 512 * 4 * 4