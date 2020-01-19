import torch
import torch.nn as nn
import torchvision

class VGG(nn.Module):
    """
    VGG model from "Very Deep Convolutional Networks for Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`.
    """

    def __init__(self, args):
        """
        Initializes the VGG model based on the provided arguments.

        Parameters
        ----------
        args: argparse.ArgumentParser
            Object that contains all the command line arguments
        """

        super(VGG, self).__init__()
        self.args = args
        if args.model == VGG:
            vgg = torchvision.models.vgg16_bn(pretrained=args.pretrained)
        else:
            vgg = torchvision.models.vgg19_bn(pretrained=args.pretrained)

        self.model = vgg.features
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))


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

        x = self.model(x)
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

        return 512 * 1 * 1