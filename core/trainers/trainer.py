from tqdm import tqdm
import torch
import torch.nn.functional as F
import copy

from core.models.classifier import Classifier
from util.general_functions import get_optimizer, make_data_loader, get_loss_function, get_class_weights, get_class_names, calculate_metrics, init_model
from util.lr_scheduler import LR_Scheduler
from util.summary import TensorboardSummary
from constants import *

class Trainer(object):
    """Helper class to train neural networks."""

    def __init__(self, args):
        """
        Creates the model, dataloader, loss function, optimizer and tensorboard summary for training.

        Parameters
        ----------
        args: argparse.ArgumentParser
            Object that contains all the command line arguments
        """

        self.args = args
        self.best_acc = 0
        self.summary = TensorboardSummary(args)
        self.model = Classifier(args)
        self.best_model = copy.deepcopy(self.model)

        if args.inference:
            self.model = self.summary.load_network(self.model)

        if not args.pretrained:
            self.model = init_model(self.model, args.init_type)
        self.optimizer = get_optimizer(self.model, args)

        if args.inference == 1:
            self.test_loader = make_data_loader(args, TEST)
        elif args.trainval:
            self.train_loader = make_data_loader(args, TRAINVAL)
        else:
            self.train_loader, self.val_loader = make_data_loader(args, TRAIN), make_data_loader(args, VAL)

        self.weights = get_class_weights(args)
        self.criterion = get_loss_function(args, self.weights)

        if not args.inference:
            self.scheduler = LR_Scheduler(args.lr_policy, args.lr, args.epochs, len(self.train_loader))

    def run_epoch(self, epoch, split=TRAIN):
        """
        Runs the model on the given dataset split for 1 full epoch.

        Parameters
        ----------
        epoch : int
            Current epoch
        split : str
            Current split
        """

        loss = 0.0
        outputs, predictions, targets = [], [], []

        if split == TRAIN:
            self.model.train()
            loader = self.train_loader
        elif split == VAL:
            self.model.eval()
            loader = self.val_loader
        else:
            self.model.eval()
            loader = self.test_loader

        bar = tqdm(loader)
        num_img = len(loader)

        for i, sample in enumerate(bar):
            with torch.autograd.set_detect_anomaly(True):
                image = sample[0]
                target = sample[1].squeeze()

                if self.args.cuda:
                    image, target = image.cuda(), target.cuda()

                if split == TRAIN:
                    self.scheduler(self.optimizer, i, epoch, self.best_acc)
                    self.optimizer.zero_grad()
                    output = self.model(image)
                else:
                    with torch.no_grad():
                        output = self.model(image)

                loss = self.criterion(output, target)
                if split == TRAIN:
                    loss.backward()

                    if self.args.clip > 0:
                        if self.args.gpu_ids:
                            torch.nn.utils.clip_grad_norm_(self.model.module().parameters(), self.args.clip)
                        else:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)

                    self.optimizer.step()

                loss += loss.item()
                bar.set_description(split +' loss: %.3f' % (loss / (i + 1)))

                outputs.extend(F.softmax(output, dim=1).detach().cpu().numpy())
                predictions.extend(torch.argmax(output, dim=1).cpu().numpy())
                targets.extend(target.cpu().numpy())

                # Show 10 * 3 inference results each epoch
                if i % (num_img // 10) == 0:
                   self.summary.visualize_image(image, F.softmax(output, dim=1), target, split=split)

        accuracy, balanced_accuracy, recall, precision, f1, roc_auc = calculate_metrics(targets, predictions, outputs)
        self.summary.add_results(epoch, loss, accuracy, balanced_accuracy, recall, precision, f1, roc_auc, split=split)

        if split == VAL and accuracy > self.best_acc:
            self.best_acc = accuracy
            self.best_model = copy.deepcopy(self.model)

        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))

    def inference(self):
        """
        Runs the model on the given test dataset

        Returns
        -------

        """

        classes = get_class_names(self.args)
        self.model.eval()
        loader = self.test_loader
        bar = tqdm(loader)

        image_paths = []
        predictions = []

        for i, sample in enumerate(bar):
            with torch.autograd.set_detect_anomaly(True):
                image = sample[0]
                image_paths += sample[1]

                if self.args.cuda:
                    image = image.cuda()

                with torch.no_grad():
                    output = self.model(image)
                    predictions.append(classes[torch.argmax(output, dim=1).cpu().item()])

        return image_paths, predictions

    def save_network(self):
        """Saves the network's parameters."""

        self.summary.save_network(self.best_model)

    def load_network(self, args):
        """
        Loads the network's parameters

        Parameters
        ----------
        args : argparse.ArgumentParser
            Object that contains all the command line arguments
        """

        self.model = Classifier(args)
        self.model.load_state_dict(torch.load(''))