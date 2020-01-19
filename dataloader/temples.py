import os
from PIL import Image
from torch.utils import data
import torchvision.transforms as standard_transforms
import itertools
import sklearn.model_selection as model_selection
import sklearn.utils.class_weight as class_weight
import numpy as np
import torch

from constants import *
import util.custom_transforms as custom_transforms

class Temples(data.Dataset):
    """Class for the temples dataset."""

    NUM_CLASSES = 11
    CLASSES = ['Armenia', 'Australia', 'Germany', 'Hungary+Slovakia+Croatia', 'Indonesia-Bali', 'Japan', 'Malaysia+Indonesia', 'Portugal+Brazil', 'Russia', 'Spain', 'Thailand']
    TRAINVAL_CLASS_WEIGHTS = [5.90082645, 1.85454545, 0.60662702, 1.32467532, 1.44242424, 1.04692082, 1.18016529, 1.2020202,  0.52346041, 0.95454545, 0.62412587]
    TRAIN_CLASS_WEIGHTS = [6.12121212, 1.66942149, 0.64059197, 1.25206612, 1.44976077, 1.00165289, 1.25206612, 1.172147, 0.50542118, 1.00165289, 0.64059197]
    ROOT = '../../'
    DATASET = 'dataset'


    def __init__(self, args, split=TRAINVAL):
        """
        Loads the relevant dataset files based on the given split.

        Parameters
        ----------
        args: argparse.ArgumentParser
            Object that contains all the command line arguments
        split : str
            Current dataset split
        """

        self.args = args

        if split == TEST:
            self.dataset = self.make_test_dataset(args.test_dir)
        else:
            self.split = TRAIN # hardcoded for retrieving the files
            self.dataset = self.make_dataset(split)
        self.split = split

        if len(self.dataset) == 0:
            raise RuntimeError('Found 0 images, please check the dataset')

        self.class_weights = self.get_class_weights()
        self.transform = self.get_transforms()

    def __len__(self):
        """
        Computes the length of the dataset.

        Returns
        -------
        int
            Dataset length
        """

        return len(self.dataset)

    def __getitem__(self, index):
        """
        Retrieves the item in the dataset at the given index.

        Parameters
        ----------
        index : int
            Item index

        Returns
        -------
        PIL Image
            Input image
        label | image_path
        """

        if self.split == TEST:
            image_path = self.dataset[index]
        else:
            image_path, label = self.dataset[index]

        image = Image.open(image_path)

        if self.transform is not None:
            image = self.transform(image)

        if self.split == TEST:
            image_path = image_path[image_path.find(self.args.test_dir) + len(self.args.test_dir):]
            return image, image_path
        else:
            label = torch.Tensor([label]).long()
            return image, label

    def get_class_weights(self):
        """
        Computes the class weights for the current dataset split.

        Returns
        -------
        list
            Class weights
        """

        y_train = [item[1] for item in self.dataset]
        class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)

        return class_weights

    def make_test_dataset(self, dir):
        """
       Loads the test dataset files.

       Parameters
       ----------
       split : str

       Returns
       -------
       list
           Contains all the dataset files
       """

        current_dir = os.path.dirname(__file__)
        dataset_path = os.path.join(current_dir, dir)
        items = os.listdir(dataset_path)
        items = [os.path.join(dataset_path, item) for item in items]

        return items

    def make_dataset(self, split):
        """
        Loads the dataset files.

        Parameters
        ----------
        split : str

        Returns
        -------
        list
            Contains all the dataset files
        """

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

        if split == TRAIN or split == VAL:
            x= [item[0] for item in items]
            y = [item[1] for item in items]
            x_train, x_val, y_train, y_val = model_selection.train_test_split(x, y, random_state=42, test_size=0.15)

            if split == TRAIN:
                items = [(x_train[i], y_train[i]) for i in range(len(x_train))]
            elif split == VAL:
                items = [(x_val[i], y_val[i]) for i in range(len(x_val))]

        return items

    def get_transforms(self):
        """
        Returns the transforms that are applied to the input.

        Returns
        -------
        torchvision.compose
            Composition of all the transformations
        """

        if self.split == TRAIN or self.split == TRAINVAL:
            transforms = [
                standard_transforms.Resize(self.args.resize),
                standard_transforms.RandomResizedCrop(size=self.args.crop_size),
                custom_transforms.RandomRotate(0.4),
                custom_transforms.RandomGaussianBlur(),
                standard_transforms.ToTensor(),
            ]
            if self.args.normalize_input:
                transforms.append(standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
            if self.args.random_erasing:
                transforms.append(standard_transforms.RandomErasing())

            transforms = standard_transforms.Compose(transforms)
        elif self.split == VAL or self.split == TEST:
            transforms = [
                standard_transforms.Resize(self.args.resize),
                standard_transforms.CenterCrop(self.args.crop_size),
                standard_transforms.ToTensor(),
            ]
            if self.args.normalize_input:
                transforms.append(standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

            transforms = standard_transforms.Compose(transforms)
        else:
            raise RuntimeError('Invalid dataset mode')

        return transforms


        

