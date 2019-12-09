import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets
from PIL import Image


class CUB200(Dataset):
    """
    Create dataset for CUB200. First 100 class for training, second 100 for testing
    """
    def __init__(self, dataset_root, train=False, transform=None):
        img_folder = dataset_root + "/cub2011/CUB_200_2011/images"
        dataset = datasets.ImageFolder(img_folder)

        # split samples to train and test set
        condition = (lambda idx: idx < len(dataset)/2) if train else (lambda idx: idx > len(dataset)/2)
        samples = [(a, b) for idx, (a, b) in enumerate(dataset.imgs) if condition(idx)]
        self.paths = [a for (a, b) in samples]
        self.targets = [b for (a, b) in samples]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img, label = self.paths[idx], self.targets[idx]
        img = Image.open(img).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label
