import torch
import torch.nn as nn
import functools

from constants import *
from core.models.backbone.pspnet import PSPNet
from core.models.backbone.vgg import VGG
from core.models.backbone.resnet import ResNet
from util.general_functions import get_num_classes


class Classifier(nn.Module):

    def __init__(self, args):
        super(Classifier, self).__init__()
        self.args = args
        self.norm_layer = self.get_norm_layer(args.norm_layer)
        self.backbone = self.get_backbone(args)
        self.feature_maps_size = self.backbone.get_feature_maps_size()
        self.pretrained = args.pretrained
        self.decoder = self.get_decoder(self.feature_maps_size, get_num_classes(args.dataset))

    def get_backbone(self, args):
        """
        Builds the backbone based on the provided arguments and returns the initialized model

            Parameters:
            args (argparse)    -- command line arguments
        """

        if VGG_16 in args.model:
            model = VGG(args, pretrained=args.pretrained)
        elif PSPNET in args.model:
            model = PSPNet(args)
        elif RESNET in args.model:
            model = ResNet(args)
        else:
            raise NotImplementedError()

        print("Built " + args.model)
        if args.cuda:
            model = model.cuda()

        return model

    def get_decoder(self, feature_maps_size, num_classes):
        layers = []
        if feature_maps_size > 4096:
            layers.append(nn.Linear(feature_maps_size, 4096))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout())

        if feature_maps_size < 4096:
            layers.append(nn.Linear(feature_maps_size, num_classes))
        else:
            layers.append(nn.Linear(4096, 4096))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout())
            layers.append(nn.Linear(4096, 1024))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout())
            layers.append(nn.Linear(1024, num_classes))

        model = nn.Sequential(*layers)

        if self.args.cuda:
            model = model.cuda()

        return model

    def get_norm_layer(self, norm_type=INSTANCE_NORM):
        """Returns a normalization layer

            Parameters:
                norm_type (str) -- the name of the normalization layer: batch | instance | none

        For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
        For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
        """

        if norm_type == BATCH_NORM:
            norm_layer = nn.BatchNorm2d
        elif norm_type == INSTANCE_NORM:
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
        elif norm_type == 'none':
            norm_layer = None
        else:
            raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
        return norm_layer

    def get_params(self, modules):
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                for p in m[1].parameters():
                    if p.requires_grad:
                        yield p

    def get_train_parameters(self, lr):
        if self.pretrained:
            train_params = [{'params': self.get_params([self.backbone]), 'lr': lr / 10},
                            {'params': self.get_params([self.decoder]), 'lr': lr}]
        else:
            train_params = [{'params': self.get_params([self.backbone, self.decoder]), 'lr': lr}]

        return train_params


    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.decoder(x)

        return x
