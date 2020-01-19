import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sklearn.metrics as metrics
import csv

from dataloader.temples import Temples
from constants import *

def make_data_loader(args, split=TRAIN):
    """
    Creates dataset loader for the given split.

    Parameters
    ----------
    args : argparse.ArgumentParser
        Object that contains all the command line arguments
    split : str

    Returns
    -------
    DataLoader
        Dataset loader
    """

    if split == TEST:
        test_set = Temples(args, split=TEST)
        loader = DataLoader(test_set, batch_size=1, num_workers=1, shuffle=False)
    elif split == TRAINVAL:
        trainval_set = Temples(args, split=TRAINVAL)
        loader = DataLoader(trainval_set, batch_size=args.batch_size, num_workers=1, shuffle=True)
    else:
        set = Temples(args, split=split)
        if split == TRAIN:
            loader = DataLoader(set, batch_size=args.batch_size, num_workers=1, shuffle=True)
        else:
            loader = DataLoader(set, batch_size=args.batch_size, num_workers=1, shuffle=False)

    return loader

def get_class_names(args):
    """
    Returns the class names

    Parameters
    ----------
    args : argparse.ArgumentParser
        Object that contains all the command line arguments

    Returns
    -------
    list
        Class names
    """

    names = None
    if args.dataset == TEMPLES_DATASET:
        names = Temples.CLASSES
    else:
        raise NotImplementedError()

    return names


def get_class_weights(args):
    """
    Computes the class weights for the dataset split.

    Parameters
    ----------
    args : argparse.ArgumentParser
        Object that contains all the command line arguments

    Returns
    -------
    list
        Class weights
    """

    weights = None
    if args.dataset == TEMPLES_DATASET:
        if args.trainval:
            weights = Temples.TRAINVAL_CLASS_WEIGHTS
        else:
            weights = Temples.TRAIN_CLASS_WEIGHTS
    else:
        raise NotImplementedError()

    return weights

def get_loss_function(args, weights):
    """
    Returns the loss function used for training.

    Parameters
    ----------
    args : argparse.ArgumentParser
        Object that contains all the command line arguments
    weights : list
        Class weights

    Returns
    -------
    torch.nn.zloss._Loss
        Loss function used for training
    """

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
    """
    Gets the dataset number of classes.

    Parameters
    ----------
    dataset : str
        Current dataset

    Returns
    -------
    int
        Number of classes
    """

    if dataset == TEMPLES_DATASET:
        num_classes = Temples.NUM_CLASSES
    else:
        raise NotImplementedError

    return num_classes

def get_optimizer(model, args):
    """
    Builds the optimizer for the model based on the provided arguments and returns the optimizer.

    Parameters
    ----------
    model : torch.nn.Module
        The network to be optimized
    args : argparse.ArgumentParser
        Object that contains all the command line arguments

    Returns
    -------
    optim
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
    """
    Initialize the network weights.

    Parameters
    ----------
    net : torch.nn.Module
        The network to be optimized
    init_type : str
        The name of an initialization method: normal | xavier | kaiming | orthogonal
    init_gain : float
        Scaling factor for normal, xavier and orthogonal

     Returns
     -------
     torch.nn.Module
     """

    init_weights(net, init_type, init_gain=init_gain)
    return net

def init_weights(net, init_type=NORMAL_INIT, init_gain=0.02):
    """
    Initialize the network weights.

    Parameters
    ----------
    net : torch.nn.Module
        The network to be optimized
    init_type : str
        The name of an initialization method: normal | xavier | kaiming | orthogonal
    init_gain : float
        Scaling factor for normal, xavier and orthogonal
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
    """
    Calculate accuracy metrics.

    Parameters
    ----------
    targets : list
        Network targets
    predictions : list
        Network predictions
    outputs : list
        Network outputs

    Returns
    -------
    accuracy : float
    balanced_accuracy : float
    recall : float
    precision : float
    f1 : float
    roc_auc float
    """

    accuracy = metrics.accuracy_score(targets, predictions)
    balanced_accuracy = metrics.balanced_accuracy_score(targets, predictions)
    recall = metrics.recall_score(targets, predictions, average='micro')
    precision = metrics.precision_score(targets, predictions, average='micro')
    f1 = metrics.f1_score(targets, predictions, average='micro')
    roc_auc = metrics.roc_auc_score(targets, outputs, multi_class='ovr')

    return accuracy, balanced_accuracy, recall, precision, f1, roc_auc

def write_csv_file(image_paths, predictions):
    with open('results.csv', mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar="'", quoting=csv.QUOTE_MINIMAL)

        for i in range(len(image_paths)):
            csv_writer.writerow([image_paths[i], predictions[i]])

def print_training_info(args):
    """
    Prints the training info to the command line.

    Parameters
    ----------
    args : argparse.ArgumentParser
        Object that contains all the command line arguments
    """

    print('Pretrained', args.pretrained)
    print('Optimizer', args.optim)
    print('Learning rate', args.lr)
    if args.clip > 0:
        print('Gradient clip', args.clip)
    print('Resize', args.resize)
    print('Weighted', args.use_class_weights)
    print('Normalizing input', args.normalize_input)
    print('Random erasing', args.random_erasing)
    print('Batch size', args.batch_size)
    print('Norm layer', args.norm_layer)
    print('Using cuda', torch.cuda.is_available())
    print('Using ' + args.loss_type + ' loss')
    print('Starting Epoch:', args.start_epoch)
    print('Total Epoches:', args.epochs)



