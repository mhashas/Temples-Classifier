import torch
import torch.nn as nn
from torchvision import models

from constants import *

class ResNet(nn.Module):
    """
    ResNet model from "Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`.
    """

    def __init__(self, args):
        """
        Initializes the ResNet based on the provided arguments.

        Parameters
        ----------
        args: argparse.ArgumentParser
            Object that contains all the command line arguments
        """

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
        """
        Computes a forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input features

        Returns:
        torch.Tensor
            Output features
        """

        x = self.backbone(x)
        x = self.avgpool(x)

        return x

    def get_feature_maps_size(self):
        """
        Flattened size of the feature maps from the last layer.

        Returns
        -------
        int
            Flattened size of the feature maps from the last layer
        """

        return 2048 * 1 * 1
