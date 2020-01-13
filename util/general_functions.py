import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import sklearn.metrics as metrics

from dataloader.temples import Temples
from constants import *

def make_data_loader(args, split=TRAIN):
    """
    Builds the model based on the provided arguments

        Parameters:
        args (argparse)    -- input arguments
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
    """

    # txt files exit?
        # load text files
    # else
        # generate txt files with train, val

    if split == TRAINVAL:
        trainval_set = Temples(args, split=TRAINVAL)
        loader = DataLoader(trainval_set, batch_size=args.batch_size, num_workers=8, shuffle=True)
    else:
        set = Temples(args, split=split)
        loader = DataLoader(set, batch_size=args.batch_size, num_workers=8, shuffle=True)

    return loader

def get_class_weights(args):
    weights = None
    if args.dataset == TEMPLES_DATASET:
        weights = Temples.CLASS_WEIGHTS
    else:
        raise NotImplementedError()

    return weights

def get_loss_function(args, weights):
    if not args.use_class_weights:
        weights = None
    else:
        weights = torch.Tensor(weights).cuda() if torch.cuda.is_available() else torch.Tensor(weights)

    if args.loss_type == CE_LOSS:
        loss = torch.nn.CrossEntropyLoss(weight=weights)
    else:
        raise NotImplementedError

    return loss

def get_num_classes(dataset):
    if dataset == TEMPLES_DATASET:
        num_classes = Temples.NUM_CLASSES
    else:
        raise NotImplementedError

    return num_classes


def get_optimizer(model, args):
    """
    Builds the optimizer for the model based on the provided arguments and returns the optimizer

        Parameters:
        model          -- the network to be optimized
        args           -- command line arguments
    """
    if args.gpu_ids:
        train_params = model.module.get_train_parameters(args.lr)
    else:
        train_params = model.get_train_parameters(args.lr)

    if args.optim == SGD:
        optimizer = optim.SGD(train_params, lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum, nesterov=True)
    elif args.optim == ADAM:
        optimizer = optim.Adam(train_params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == AMSGRAD:
        optimizer = optim.Adam(train_params, lr=args.lr, weight_decay=args.weight_decay, amsgrad=True)
    else:
        raise NotImplementedError

    return optimizer

def init_model(net, init_type=NORMAL_INIT, init_gain=0.02):
    """Initialize the network weights

    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.

    Return an initialized network.
    """

    init_weights(net, init_type, init_gain=init_gain)
    return net

def init_weights(net, init_type=NORMAL_INIT, init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == NORMAL_INIT:
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == XAVIER_INIT:
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == KAIMING_INIT:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='leaky_relu')
            elif init_type == ORTHOGONAL_INIT:
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif hasattr(m, '_all_weights') and (classname.find('LSTM') != -1 or classname.find('GRU') != -1):
            for names in m._all_weights:
                for name in filter(lambda n: "weight" in n, names):
                    weight = getattr(m, name)
                    nn.init.xavier_normal_(weight.data, gain=init_gain)

                for name in filter(lambda n: "bias" in n, names):
                    bias = getattr(m, name)
                    nn.init.constant_(bias.data, 0.0)

                    if classname.find('LSTM') != -1:
                        n = bias.size(0)
                        start, end = n // 4, n // 2
                        nn.init.constant_(bias.data[start:end], 1.)
        elif classname.find('BatchNorm2d') != -1 or classname.find('SynchronizedBatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    print('Initialized network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

def calculate_metrics(targets, predictions, outputs):
    accuracy = metrics.accuracy_score(targets, predictions)
    balanced_accuracy = metrics.balanced_accuracy_score(targets, predictions)
    recall = metrics.recall_score(targets, predictions, average='micro')
    precision = metrics.precision_score(targets, predictions, average='micro')
    f1 = metrics.f1_score(targets, predictions, average='micro')
    roc_auc = metrics.roc_auc_score(targets, outputs, multi_class='ovr')

    return accuracy, balanced_accuracy, recall, precision, f1, roc_auc


def tensor2im(input_image, imtype=np.uint8, return_tensor=True):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor.cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.ndim == 3:
            image_numpy = (image_numpy - np.min(image_numpy))/(np.max(image_numpy)-np.min(image_numpy))
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (image_numpy + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image

    return torch.from_numpy(image_numpy.astype(imtype)) if return_tensor else np.transpose(image_numpy, (1,2,0))

def print_training_info(args):
    if 'unet' in args.model:
        print('Ngf', args.ngf)
        print('Num downs', args.num_downs)
        print('Down type', args.down_type)

    if 'deeplab' in args.model:
        print('Pretrained', args.pretrained)
        print('Output stride', args.output_stride)

    print('Optimizer', args.optim)
    print('Learning rate', args.lr)

    if args.clip > 0:
        print('Gradient clip', args.clip)

    print('Resize', args.resize)
    print('Batch size', args.batch_size)
    print('Norm layer', args.norm_layer)
    print('Using cuda', torch.cuda.is_available())
    print('Using ' + args.loss_type + ' loss')
    print('Starting Epoch:', args.start_epoch)
    print('Total Epoches:', args.epochs)



