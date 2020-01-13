import os
import torch
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
import glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from dataloader.temples import Temples

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

    def add_results(self, epoch, loss, accuracy, balanced_accuracy, recall, precision, f1, roc_auc, split="train"):
        self.writer.add_scalar(split + '/loss_epoch', loss, epoch)
        self.writer.add_scalar(split + '/acc', accuracy, epoch)
        self.writer.add_scalar(split + '/balanced_accuracy', balanced_accuracy, epoch)
        self.writer.add_scalar(split + '/recall', recall, epoch)
        self.writer.add_scalar(split + '/precision', precision, epoch)
        self.writer.add_scalar(split + '/f1', f1, epoch)
        self.writer.add_scalar(split + '/roc_auc', roc_auc, epoch)

    def matplotlib_imshow(self, img, one_channel=False):
        if one_channel:
            img = img.mean(dim=0)
        npimg = img
        if one_channel:
            plt.imshow(npimg, cmap="Greys")
        else:
            plt.imshow(np.transpose(npimg, (1, 2, 0)))

    def visualize_image(self, images, outputs, targets, split="train"):
        targets = targets.cpu().numpy()
        images = images.cpu().numpy()
        fig = plt.figure(figsize=(12, 12))
        number_images = 4
        number_rows = int(images.shape[0] / 4)

        for row in np.arange(number_rows):
            for idx in np.arange(number_images):
                ax = fig.add_subplot(row + 1, number_images, idx + 1, xticks=[], yticks=[])

                self.matplotlib_imshow(images[idx+row*4], one_channel=False)
                ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
                    Temples.CLASSES[torch.argmax(outputs[idx+row*4], dim=0)],
                    max(outputs[idx+row*4].detach().cpu().numpy()) * 100.0,
                    Temples.CLASSES[targets[idx+row*4]]),
                    color=("green" if torch.argmax(outputs[idx+row*4], dim=0) == targets[idx+row*4].item() else "red"))

        step = self.get_step(split)
        self.writer.add_figure(split + '/ZZ Image', fig, step)

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
