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

    user_groups = mnist_noniid_unequal(train_dataset, args['num_users'])

    return train_dataset, test_dataset, user_groups

def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg
