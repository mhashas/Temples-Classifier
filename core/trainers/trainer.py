from tqdm import tqdm
import torch
import torch.nn.functional as F

from core.models.classifier import Classifier
from util.general_functions import get_optimizer, make_data_loader, get_loss_function, get_class_weights, calculate_metrics
from util.lr_scheduler import LR_Scheduler
from util.summary import TensorboardSummary
from constants import *

class Trainer(object):

    def __init__(self, args):
        self.args = args
        self.best_acc = 0

        self.model = Classifier(args)
        self.optimizer = get_optimizer(self.model, args)
        self.summary = TensorboardSummary(args)
        self.weights = get_class_weights(args)

        if args.trainval:
            self.train_loader = make_data_loader(args, TRAINVAL)
        else:
            self.train_loader, self.test_loader = make_data_loader(args, TRAIN)

        self.criterion = get_loss_function(args, self.weights)
        self.scheduler = LR_Scheduler(args.lr_policy, args.lr, args.epochs, len(self.train_loader))

    def run_epoch(self, epoch, split=TRAIN):
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

                self.scheduler(self.optimizer, i, epoch, self.best_acc)

                if split == TRAIN:
                    self.optimizer.zero_grad()
                    output = self.model(image)
                else:
                    with torch.no_grad():
                        output = self.model(image)

                loss = self.criterion(output, target)

                # Show 10 * 3 inference results each epoch
                if i % (num_img // 10) == 0:
                    self.summary.visualize_image(image, target, output, split=split)

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

        accuracy, balanced_accuracy, recall, precision, f1, roc_auc = calculate_metrics(targets, predictions, outputs)
        self.summary.add_results(loss, accuracy, balanced_accuracy, recall, precision, f1, roc_auc)

        if accuracy < self.best_acc:
            self.best_acc = accuracy

        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))


    def save_network(self):
        self.summary.save_network(self.model)

    def load_network(self, args):
        self.model = Classifier(args)
        self.model.load_state_dict(torch.load(''))