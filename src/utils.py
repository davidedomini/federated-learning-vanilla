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
                                   transform=apply_transform)

    test_dataset = datasets.MNIST(data_dir,
                                  train=False,
                                  download=True,
                                  transform=apply_transform)

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

def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return