import os
import torch
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
import glob
import numpy as np

from util.general_functions import tensor2im

class TensorboardSummary(object):

    def __init__(self, args):
        self.args = args
        self.experiment_dir = self.generate_directory(args)
        self.writer = SummaryWriter(log_dir=os.path.join(self.experiment_dir))

        self.train_step = 0
        self.test_step = 0

    def generate_directory(self, args):
        checkname = 'debug' if args.debug else ''
        checkname += args.model

        if 'deeplab' in args.model:
            checkname += '-pretrained' if args.pretrained else ''
            checkname += '-os_' + str(args.output_stride)

        if 'unet' in args.model:
            checkname += '-numdowns_' + str(args.num_downs) + '-ngf_' + str(args.ngf) + '-downtype_' + str(args.down_type)

        checkname += '-inittype_' + args.init_type
        checkname += '-optim_' + args.optim + '-lr_' + str(args.lr)

        if args.clip > 0:
            checkname += '-clipping_' + str(args.clip)

        if args.resize:
            checkname += '-resize_' + ','.join([str(x) for x in list(args.resize)])
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
        self.writer.add_scalar(tag, value, step)

    def add_results(self, loss, accuracy, balanced_accuracy, recall, precision, f1, roc_auc, split="train"):
        step = self.get_step(split)
        self.writer.add_scalar(split + '/loss_epoch', loss, step)
        self.writer.add_scalar(split + '/acc', accuracy, step)
        self.writer.add_scalar(split + '/balanced_accuracy', balanced_accuracy, step)
        self.writer.add_scalar(split + '/recall', recall, step)
        self.writer.add_scalar(split + '/precision', precision, step)
        self.writer.add_scalar(split + '/f1', f1, step)
        self.writer.add_scalar(split + '/roc_auc', roc_auc, step)

    def visualize_image(self, image, target, output=None, split="train"):
        step = self.get_step(split)
        images = []
        outputs = []
        targets = []

        number_of_images = min(5, image.size(0))

        for i in range(number_of_images):
            current_image = image[i]
            images.append(tensor2im(current_image))

        grid_image = make_grid(images)
        self.writer.add_image(split + '/ZZ Image', grid_image, step)

        ## @TODO RADU check how to visualize correct class and incorrect class nicely in tensorboard !!

    def save_network(self, model):
        path = self.args.save_dir + '/' + self.experiment_dir.replace('./', '')

        if not os.path.isdir(path):
            os.makedirs(path)

        torch.save(model.state_dict(), path + '/' + self.args.model + '_pretrained:' + str(self.args.pretrained) + '.pth')

    def get_step(self, split):
        if split == 'train':
            self.train_step += 1
            return self.train_step
        elif split == 'test':
            self.test_step += 1
            return self.test_step
