import copy
import torch
from torchvision import datasets, transforms
from sampling import mnist_noniid_unequal


def get_dataset(args):
    data_dir = '../data/mnist/'
    apply_transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(data_dir,
                                   train=True,
                                   download=True,
                                   transforms=apply_transform)

    test_dataset = datasets.MNIST(data_dir,
                                  train=False,
                                  download=True,
                                  transforms=apply_transform)

    user_groups = mnist_noniid_unequal(train_dataset, args.num_users)

    return train_dataset, test_dataset, user_groups
