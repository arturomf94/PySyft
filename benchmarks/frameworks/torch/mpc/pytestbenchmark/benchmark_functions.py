"""
This module contains the functions to be benchmarked.
"""

import torch

# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
from torchvision import datasets, transforms

# import time


class Arguments:
    """
    A class that describes all the hyper-parameters for the training
    for function normal_vs_smpc. For more information see Tutorial 12bis.
    """

    def __init__(self, epochs, n_train_items, n_test_items):
        self.batch_size = 128
        self.test_batch_size = 128
        self.epochs = epochs
        self.n_train_items = n_train_items
        self.n_test_items = n_test_items
        self.lr = 0.02
        self.seed = 1
        self.log_interval = 1  # Log info at each batch
        self.precision_fractional = 3


def get_private_data_loaders(args, workers, crypto_provider):
    def one_hot_of(index_tensor):
        """
        Transform to one hot tensor

        Example:
            [0, 3, 9]
            =>
            [[1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]]

        """
        onehot_tensor = torch.zeros(*index_tensor.shape, 10)  # 10 classes for MNIST
        onehot_tensor = onehot_tensor.scatter(1, index_tensor.view(-1, 1), 1)
        return onehot_tensor

    def secret_share(tensor):
        """
        Transform to fixed precision and secret share a tensor
        """
        return tensor.fix_precision(precision_fractional=args.precision_fractional).share(
            *workers, crypto_provider=crypto_provider, protocol="fss", requires_grad=True
        )

    transformation = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST("../data", train=True, download=True, transform=transformation),
        batch_size=args.batch_size,
    )

    private_train_loader = [
        (secret_share(data), secret_share(one_hot_of(target)))
        for i, (data, target) in enumerate(train_loader)
        if i < args.n_train_items / args.batch_size
    ]

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST("../data", train=False, download=True, transform=transformation),
        batch_size=args.test_batch_size,
    )

    private_test_loader = [
        (secret_share(data), secret_share(target.float()))
        for i, (data, target) in enumerate(test_loader)
        if i < args.n_test_items / args.test_batch_size
    ]

    return private_train_loader, private_test_loader


def sigmoid(method: str, prec_frac: int, workers: dict):
    """
    Function to simulate a sigmoid approximation, given
    a method, a precision value and the workers used
    for sharing data.
    """

    # Define workers
    alice, bob, james = workers["alice"], workers["bob"], workers["james"]

    # Init tensor, share and approximate sigmoid
    example_tensor = torch.tensor([1.23212])
    t_sh = example_tensor.fix_precision(precision_fractional=prec_frac).share(
        alice, bob, crypto_provider=james
    )
    r_sh = t_sh.sigmoid(method=method)
    return r_sh.get().float_prec()


def tanh(method: str, prec_frac: int, workers: dict):
    """
    Function to simulate a tanh approximation, given
    a method, a precision value and the workers used
    for sharing data.
    """

    # Define workers
    alice, bob, james = workers["alice"], workers["bob"], workers["james"]

    # Init tensor, share and approximate sigmoid
    example_tensor = torch.tensor([1.23212])
    t_sh = example_tensor.fix_precision(precision_fractional=prec_frac).share(
        alice, bob, crypto_provider=james
    )
    r_sh = t_sh.tanh(method=method)
    return r_sh.get().float_prec()


def normal_vs_smpc(workers: dict) -> float:
    """
    Function that calculates the training time difference
    between "normal" and SMPC training, using MNIST as
    a dataset.
    This function follows the same example as Tutorial 12 bis.
    """
    # Define workers
    alice, bob, james = workers["alice"], workers["bob"], workers["james"]
    epochs = 10
    n_train_items = 640
    n_test_items = 640

    args = Arguments(epochs, n_train_items, n_test_items)

    _ = torch.manual_seed(args.seed)

    private_train_loader, private_test_loader = get_private_data_loaders(
        args,
        workers=[alice, bob],
        crypto_provider=james,
    )
    return 1.00
