import os
from PIL import Image
from torch.utils import data
import torchvision.transforms as standard_transforms
import itertools
import sklearn
import numpy as np
import torch

from constants import *
import util.custom_transforms as custom_transforms

class Temples(data.Dataset):

    NUM_CLASSES = 11
    CLASSES = ['Armenia', 'Australia', 'Germany', 'Hungary+Slovakia+Croatia', 'Indonesia-Bali', 'Japan', 'Malaysia+Indonesia', 'Portugal+Brazil', 'Russia', 'Spain', 'Thailand']
    CLASS_WEIGHTS = [5.90082645, 1.85454545, 0.60662702, 1.32467532, 1.44242424, 1.04692082, 1.18016529, 1.2020202,  0.52346041, 0.95454545, 0.62412587]
    ROOT = '../../'
    DATASET = 'dataset'


    def __init__(self, args, split=TRAINVAL):
        self.args = args
        self.split = split

        if split == TRAINVAL:
            self.dataset = self.make_dataset()
        else:
            raise NotImplementedError()
            self.dataset = self.make_dataset()

        if len(self.dataset) == 0:
            raise RuntimeError('Found 0 images, please check the dataset')

        self.class_weights = self.get_class_weights()
        self.transform = self.get_transforms()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image_path, label = self.dataset[index]
        image = Image.open(image_path)

        if self.transform is not None:
            image = self.transform(image)

        label = torch.Tensor([label]).long()

        return image, label

    def make_dataset_from_file(self, file):
        items = []

        # @TODO read split from file

        return items

    def get_class_weights(self):
        y_train = [item[1] for item in self.dataset]
        class_weights = sklearn.utils.class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)

        return class_weights

    def make_dataset(self):
        current_dir = os.path.dirname(__file__)
        dataset_path = os.path.join(current_dir, self.ROOT, self.DATASET, self.split)


        classes = os.listdir(dataset_path)
        items = []

        for i in range(len(classes)):
            label = i
            images_path = os.path.join(dataset_path, classes[i])
            images = os.listdir(images_path)
            class_items = [(os.path.join(images_path, image), label) for image in images]
            items.append(class_items)

        items = list(itertools.chain.from_iterable(items))
        return items

    def get_transforms(self):
        if self.split == TRAIN or self.split == TRAINVAL:
            transforms = standard_transforms.Compose([
                standard_transforms.RandomResizedCrop(size=self.args.resize),
                #standard_transforms.Resize(self.args.resize),
                standard_transforms.RandomRotation(degrees=180),
                custom_transforms.RandomGaussianBlur(),
                standard_transforms.ToTensor(),
                #standard_transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                standard_transforms.RandomErasing(),

            ])
        elif self.split == VAL or self.split == TEST:
            transforms = standard_transforms.Compose([
                standard_transforms.Resize(self.args.resize),
                standard_transforms.ToTensor(),
                # standard_transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            raise RuntimeError('Invalid dataset mode')

        return transforms


        

