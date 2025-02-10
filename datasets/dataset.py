import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import datasets


class CustomDataset(Dataset):
    def __init__(self, root="./Dataset", train=True, transform=None, test_transform=None, download_dataset=False):
        super().__init__()

        self.root = os.path.join(root, "CIFAR10")

        if not os.path.exists(self.root):
            os.makedirs(self.root)

        if download_dataset:
            self.cifar10_train = datasets.CIFAR10(root=self.root, train=True, download=True)
            self.cifar10_test = datasets.CIFAR10(root=self.root, train=False, download=True)
        else:
            self.cifar10_train = datasets.CIFAR10(root=self.root, train=True, download=False)
            self.cifar10_test = datasets.CIFAR10(root=self.root, train=False, download=False)
        

        self.transform = transform
        self.test_transform = test_transform

    def __len__(self):
        if train:
            return len(self.cifar10_train)
        else:
            return len(self.cifar10_test)

    def __getitem__(self, idx):
        if train:
            image, label = self.cifar10_train[idx]
            if self.transform:
                image = self.transform(image)
            return image, label
        else:
            image, label = self.cifar10_test[idx]
            if self.test_transform:
                image = self.test_transform(image)
            return image, label
