import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

from constants import *

RESNET_101 = 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'

class _PyramidPoolingModule(nn.Module):
    """Creates a pyramid pooling module"""

    def __init__(self, in_dim, reduction_dim, setting):
        """Initializes the pyramid pooling module"""
        super(_PyramidPoolingModule, self).__init__()
        self.features = []
        for s in setting:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(s),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim, momentum=.95),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

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

        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        out = torch.cat(out, 1)
        return out


class PSPNet(nn.Module):
    """
    PSPNet model from "Pyramid Scene Parsing Network" <https://arxiv.org/pdf/1409.1556.pdf>`.
    """

    def __init__(self, args):
        """
         Initializes the PSPNet model based on the provided arguments.

         Parameters
         ----------
         args: argparse.ArgumentParser
             Object that contains all the command line arguments
         """

        super(PSPNet, self).__init__()
        self.args = args

        if args.model == PSPNET:
            resnet = models.resnet101(pretrained=args.pretrained)
        else:
            resnet = models.resnet50(pretrained=args.pretrained)

        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.backbone = nn.Sequential(self.layer0, resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        self.ppm = _PyramidPoolingModule(2048, 512, (1, 2, 3, 6))

        self.final = nn.Sequential(
            nn.Conv2d(4096, 2048, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(2048, momentum=.95),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
            #nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            #nn.BatchNorm2d(256, momentum=.95),
            #nn.ReLU()
        )

        self.initialize_weights(self.ppm, self.final)


    def initialize_weights(self, *models):
        """
         Initializes the weights of the provided modules.

         Parameters
         ----------
         models: list of torch.nn.Module
             Modules to initialize weights for
         """

        for model in models:
            for module in model.modules():
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(module.weight)
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, nn.BatchNorm2d):
                    module.weight.data.fill_(1)
                    module.bias.data.zero_()

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
        x = self.ppm(x)
        x = self.final(x)

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