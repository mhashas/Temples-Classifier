import argparse
import torch
import os
import math

from constants import *

class ParserOptions():
    """This class defines options that are used by the program"""

    def __init__(self):
        parser = argparse.ArgumentParser(description='PyTorch Semantic Video Segmentation training')

        # model specific
        parser.add_argument('--model', type=str, default=RESNET_50, choices=[RESNET, RESNET_50, PSPNET, PSPNET_50, VGG_16, VGG_19], help='model name (default:' + RESNET + ')')
        parser.add_argument('--dataset', type=str, default=TEMPLES_DATASET, choices=[TEMPLES_DATASET], help='dataset name (default:' + TEMPLES_DATASET + ')')
        parser.add_argument('--loss_type', type=str, default=CE_LOSS, choices=[CE_LOSS], help='loss func type (default:' + CE_LOSS + ')')
        parser.add_argument('--use_class_weights', type=int, default=1, choices=[0,1], help='if we should use class weights to deal with imbalanced dataset')
        parser.add_argument('--norm_layer', type=str, default=BATCH_NORM, choices=[INSTANCE_NORM, BATCH_NORM, SYNC_BATCH_NORM])
        parser.add_argument('--init_type', type=str, default=NORMAL_INIT, choices=[NORMAL_INIT, KAIMING_INIT, XAVIER_INIT, ORTHOGONAL_INIT])
        parser.add_argument('--batch_size', type=int, default=8, metavar='N', help='input batch size for training (default: 2)')
        parser.add_argument('--optim', type=str, default=ADAM, choices=[SGD, ADAM, RMSPROP, AMSGRAD, ADABOUND])
        parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate (default: auto)')
        parser.add_argument('--lr_policy', type=str, default='poly', choices=['poly', 'step', 'cos', 'linear'], help='lr scheduler mode: (default: poly)')
        parser.add_argument('--weight-decay', type=float, default=5e-4, metavar='M', help='w-decay (default: 5e-4)')
        parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='momentum (default: 0.9)')
        parser.add_argument('--clip', type=float, default=0, help='gradient clip, 0 means no clip (default: 0)')

        # training specific
        parser.add_argument('--resize', type=str, default='256,256', help='image resize: h,w')
        parser.add_argument('--crop_size', type=str, default='224,224', help='image crop size: h,w')
        parser.add_argument('--start_epoch', type=int, default=0, metavar='N', help='starting epoch')
        parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train (default: auto)')
        parser.add_argument('--eval-interval', type=int, default=1, help='evaluation interval (default: 1)')
        parser.add_argument('--trainval', type=int, default=0, choices=[0,1], help='determines whether whe should use validation images as well for training')
        parser.add_argument('--debug', type=int, default=0)
        parser.add_argument('--results_root', type=str, default='..')
        parser.add_argument('--results_dir', type=str, default='results_good_val', help='models are saved here')
        parser.add_argument('--save_dir', type=str, default='saved_models')

        parser.add_argument('--pretrained', type=int, default=1, choices=[0,1], help='if we should use pretrained network or not')
        parser.add_argument('--normalize_input', type=int, default=0, choices=[0,1], help='if we should normalize the images with the imagenet mean and std')
        parser.add_argument('--random_erasing', type=int, default=0, choices=[0,1], help='if we should use random erasing as part of our preprocessing')
        args = parser.parse_args()

        if args.debug:
            args.results_dir = 'results_dummy'

        args.resize = tuple([int(x) for x in args.resize.split(',')])
        args.crop_size = tuple([int(x) for x in args.crop_size.split(',')])
        args.cuda = torch.cuda.is_available()
        args.gpu_ids = os.environ['CUDA_VISIBLE_DEVICES'] if ('CUDA_VISIBLE_DEVICES' in os.environ) else ''
        args.gpu_ids = list(range(len(args.gpu_ids.split(',')))) if (',' in args.gpu_ids and args.cuda) else None

        if args.gpu_ids and args.norm_layer == BATCH_NORM:
            args.norm_layer = SYNC_BATCH_NORM

        self.args = args

    def parse(self):
        return self.args