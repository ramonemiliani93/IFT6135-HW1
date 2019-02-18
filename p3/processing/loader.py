import os
import re
import sys

import numpy as np
from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# Imported for testing
from utils.classes import Params

# Transforms to be used when defining loaders
train_transformer = transforms.Compose([
    transforms.RandomAffine(20),
    transforms.RandomGrayscale(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(54),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.489, 0.454, 0.416), (0.251, 0.244, 0.246))])

eval_transformer = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.489, 0.454, 0.416), (0.251, 0.244, 0.246))])


class CatsDogsDataset(Dataset):
    """Cats vs Dogs dataset."""

    def __init__(self, paths, indices=None, transform=None):
        """
        Args:
            paths (dict): dictionary with paths to images.
            indices (list): list of valid indices to use for images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.paths = paths
        # Convert to dictionary for faster indexing
        self.indices = dict(zip(range(len(indices)), indices)) if indices is not None else indices
        self.transform = transform

    def __len__(self):
        """Return length of indices if provided else all the paths will be used"""
        return len(self.indices) if self.indices is not None else len(self.paths)

    def __getitem__(self, idx):
        """"""
        if self.indices is not None:
            idx = self.indices.get(idx)

        path = self.paths.get(idx)
        image = Image.open(path).convert('RGB')
        label = 0 if 'cat' in path.lower() else 1

        if self.transform:
            image = self.transform(image)

        return image, label


def get_dict_images(path, sorted=False):

    list_of_files = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        list_of_files += [os.path.join(dirpath, file) for file in filenames if file.endswith('.jpg')]
    if sorted:
        list_of_files.sort(key=lambda x: float(re.findall('(\d{1,5}).jpg', x)[0]))
    return dict(zip(range(len(list_of_files)), list_of_files))


def fetch_dataloader(type, params, image_dir):
    """Fetches the dataloader for the desired data split.

    Parameters
    ----------
    type : str
        Split of the data to get the data loader for.
    params : object
        Params object containing configuration information.
    image_dir : str
        Path to the directory containing the Images.
    Returns
    -------
    dataloaders : dict
        Dictionary containing the dataloaders fot the desired split.
    """
    assert type in ['train', 'test'], "Invalid dataloader type"
    assert ((params.ratio >= 0) and (params.ratio <= 1)), "Split must be between 0 and 1"

    dataloaders = {}

    # Return test data loader is split is test (using eval_transformer) else return dictionary containing data loaders
    # for the train and validation sets.
    if type == 'test':
        paths = get_dict_images(image_dir, sorted=True)
        test_dataloader = DataLoader(CatsDogsDataset(paths, transform=eval_transformer), batch_size=params.batch_size, 
                                     shuffle=False, num_workers=params.num_workers, pin_memory=params.cuda)

        dataloaders.update({
            'test': test_dataloader
        })

    else:
        paths = get_dict_images(image_dir)

        length_train = len(paths)
        indices = list(range(length_train))
        split = int(params.ratio * length_train)

        if params.shuffle:
            np.random.seed(params.random_seed)
            np.random.shuffle(indices)

        train_indices, val_indices = indices[split:], indices[:split]
        train_dataset = CatsDogsDataset(paths, indices=train_indices, transform=train_transformer)
        val_dataset = CatsDogsDataset(paths, indices=val_indices, transform=eval_transformer)

        train_dataloader = DataLoader(train_dataset, batch_size=params.batch_size,
                                      num_workers=params.num_workers, pin_memory=params.cuda)
        val_dataloader = DataLoader(val_dataset, batch_size=params.batch_size,
                                    num_workers=params.num_workers, pin_memory=params.cuda)

        dataloaders.update({
            'train': train_dataloader,
            'val': val_dataloader
        })

    return dataloaders


if __name__ == "__main__":
    """Simple test to call the data loaders and obtain some image plots and general statistics (mean and std)."""

    print("Starting module test", file=sys.stderr)
    json_path = '/Users/ramon/Documents/UdeM/2019/Winter/IFT6135/Homeworks/HW1/p3/experiments/base_model/params.json'
    params = Params(json_path)
    params.ratio = 0.001

    images_train = '/Users/ramon/Documents/UdeM/2019/Winter/IFT6135/Homeworks/HW1/p3/data/train/trainset'
    train_dataloader = fetch_dataloader('train', params, images_train)

    r, g, b, n = 0, 0, 0, 0

    for i, (train_batch, _) in enumerate(train_dataloader.get('train')):
        rgb = train_batch.transpose(0, 1).contiguous().view(train_batch.transpose(0, 1).shape[0], -1).mean(1)
        bs = train_batch.shape[0] * train_batch.shape[2] * train_batch.shape[3]
        r += rgb[0] * bs
        g += rgb[1] * bs
        b += rgb[2] * bs
        n += bs

    R_mean = torch.div(r, n)
    G_mean = torch.div(g, n)
    B_mean = torch.div(b, n)

    print("R pixel mean: {}".format(R_mean))
    print("G pixel mean: {}".format(G_mean))
    print("B pixel mean: {}".format(B_mean))

    r_std, g_std, b_std, n = 0, 0, 0, 0

    for i, (train_batch, _) in enumerate(train_dataloader.get('train')):
        rgb = train_batch.transpose(0, 1).contiguous().view(train_batch.transpose(0, 1).shape[0], -1).var(1)
        bs = train_batch.shape[0] * train_batch.shape[2] * train_batch.shape[3]
        r_std += rgb[0] * bs
        g_std += rgb[1] * bs
        b_std += rgb[2] * bs
        n += bs

    r_std = torch.div(r_std, n)**0.5
    g_std = torch.div(g_std, n)**0.5
    b_std = torch.div(b_std, n)**0.5

    print("R pixel std: {}".format(r_std))
    print("G pixel std: {}".format(g_std))
    print("B pixel std: {}".format(b_std))
