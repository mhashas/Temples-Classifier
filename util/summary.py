import os
import torch
from torch.utils.tensorboard import SummaryWriter
import glob
import numpy as np
import matplotlib.pyplot as plt

from dataloader.temples import Temples
from constants import *

class TensorboardSummary(object):
    """
    This class is used in order to save useful information for visualizing a model's progress through training.

    Attributes
    ----------
    args : argparse.ArgumentParser
        Object that contains all the command line arguments

    Methods
    -------
    generate_directory(args)
        Generates the name of the folder where the training information will be saved
    add_scalar(tag, value , step)
        Adds a scalar value to tensorboard results file
    add_results(epoch, loss, accuracy, balanced_accuracy, recall, precision, f1, roc_auc, split)
        Adds epoch results to tensorboard results file
    visualize_image(images, outputs, targets, split="train")
        Saves visualization imagse to tensorboard file
    save_network(model)
        Saves network parameters
    get_step(split)
        Gets current split global step.
    """

    def __init__(self, args):
        """
        Initializes the tensorboard summary object.

        Parameters
        ----------
        args: argparse.ArgumentParser
            Object that contains all the command line arguments
        """

        self.args = args
        self.experiment_dir = self.generate_directory(args)
        self.writer = SummaryWriter(log_dir=os.path.join(self.experiment_dir))

        self.train_step = 0
        self.val_step = 0

    def generate_directory(self, args):
        """
        Generates the name of the folder where the training information will be saved.

        Parameters
        ----------
        args : argparse.ArgumentParser
            Object that contains all the command line arguments

        Returns
        -------
        string
            the name of the folder where the training information will be saved
        """

        checkname = 'debug' if args.debug else ''
        checkname += args.model
        checkname += '-pretrained' if args.pretrained else '-inittype_' + args.init_type
        checkname += '-normed' if args.normalize_input else ''
        checkname += '-re' if args.random_erasing else ''
        checkname += '-optim_' + args.optim + '-lr_' + str(args.lr)
        checkname += '-weighted' if args.use_class_weights else ''
        checkname += '-resize_' + ','.join([str(x) for x in list(args.resize)])

        if args.clip > 0:
            checkname += '-clipping_' + str(args.clip)

        checkname += '-epochs_' + str(args.epochs)
        checkname += '-trainval' if args.trainval else ''

        current_dir = os.path.dirname(__file__)
        directory = os.path.join(current_dir, args.results_root, args.results_dir, args.dataset, args.model, checkname)

        runs = sorted(glob.glob(os.path.join(directory, 'experiment_*')))
        run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0
        experiment_dir = os.path.join(directory, 'experiment_{}'.format(str(run_id)))

        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir)

        return experiment_dir

    def add_scalar(self, tag, value, step):
        """
        Adds a scalar value to tensorboard results file.

        Parameters
        ----------
        tag : string
            Data identifier
        value : int
            Value to record
        step : int
            Global step value to record
        """

        self.writer.add_scalar(tag, value, step)

    def add_results(self, epoch, loss, accuracy, balanced_accuracy, recall, precision, f1, roc_auc, split="train"):
        """
        Adds epoch results to tensorboard results file.

        Parameters
        ----------
        epoch : int
        loss : float
        accuracy : float
        balanced_accuracy : float
        recall : float
        precision : float
        f1 : float
        roc_auc : float
        split : string
        """

        self.writer.add_scalar(split + '/loss_epoch', loss, epoch)
        self.writer.add_scalar(split + '/acc', accuracy, epoch)
        self.writer.add_scalar(split + '/balanced_accuracy', balanced_accuracy, epoch)
        self.writer.add_scalar(split + '/recall', recall, epoch)
        self.writer.add_scalar(split + '/precision', precision, epoch)
        self.writer.add_scalar(split + '/f1', f1, epoch)
        self.writer.add_scalar(split + '/roc_auc', roc_auc, epoch)

    def visualize_image(self, images, outputs, targets, split="train"):
        """
        Saves visualization imagse to tensorboard file.

        Parameters
        ----------
        images : torch.Tensor
        outputs : torch.Tensor
        targets : torch.Tensor
        split : string
        """

        targets = targets.cpu().numpy()
        images = images.cpu().numpy()
        fig = plt.figure(figsize=(12, 12))
        number_images = 4
        number_rows = int(images.shape[0] / 4)

        for row in np.arange(number_rows):
            for idx in np.arange(number_images):
                ax = fig.add_subplot(row + 1, number_images, idx + 1, xticks=[], yticks=[])

                plt.imshow(np.transpose(images[idx+row*4], (1, 2, 0)))
                ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
                    Temples.CLASSES[torch.argmax(outputs[idx+row*4], dim=0)],
                    max(outputs[idx+row*4].detach().cpu().numpy()) * 100.0,
                    Temples.CLASSES[targets[idx+row*4]]),
                    color=("green" if torch.argmax(outputs[idx+row*4], dim=0) == targets[idx+row*4].item() else "red"))

        step = self.get_step(split)
        self.writer.add_figure(split + '/ZZ Image', fig, step)

    def save_network(self, model):
        """
        Saves network parameters.

        Parameters
        ----------
        model : torch.nn.Module
            Neural network model
        """

        path = os.path.join(self.args.save_dir, self.experiment_dir[self.experiment_dir.find(self.args.results_dir):])

        if not os.path.isdir(path):
            os.makedirs(path)

        torch.save(model.state_dict(), path + '/model.pth')

    def load_network(self, model):
        """
        Saves network parameters.

        Parameters
        ----------
        model : torch.nn.Module
            Neural network model

        Returns
        -------
        model : torch.nnModule
            Trained neural network model
        """
        path = os.path.join(self.args.pretrained_models_dir)
        state_dict = torch.load(path + '/' + self.args.model + '.pth')
        model.load_state_dict(state_dict)

        return model

    def get_step(self, split):
        """
        Gets current split global step.

        Parameters
        ----------
        split : string
            Current training split. either: TRAIN|TEST|VAL|TRAINVAL

        Returns
        -------
        int
            Current split global step
        """

        if split == TRAIN:
            self.train_step += 1
            return self.train_step
        elif split == VAL:
            self.val_step += 1
            return self.val_step
