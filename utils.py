import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import ImageOps
from torchvision import datasets, transforms

import activations
import models
import visualize

AFS = list(activations.__class_dict__.keys())
MODELS = list(models.__class_dict__.keys())


def _colorize_grayscale_image(image):
    return ImageOps.colorize(image, (0, 0, 0), (255, 255, 255))


_SVHN_TRAIN_TRANSFORMS = _SVHN_TEST_TRANSFORMS = [
    transforms.ToTensor(),
    transforms.ToPILImage(),
    transforms.CenterCrop(28),
    transforms.ToTensor(),
]

_MNIST_COLORIZED_TRAIN_TRANSFORMS = _MNIST_COLORIZED_TEST_TRANSFORMS = [
    transforms.ToTensor(),
    transforms.ToPILImage(),
    transforms.Lambda(lambda x: _colorize_grayscale_image(x)),
    transforms.ToTensor(),
]

_DATASET_CHANNELS = {
    "MNIST": 1,
    "SVHN": 3,
    "EMNIST": 1,
    "KMNIST": 1,
    "QMNIST": 1,
    "FashionMNIST": 1
}


def get_loader(args):
    if args.exname == "AFS":
        # Load train and test data directly.
        if args.dataset == "MNIST":
            train_dataset = datasets.MNIST(
                root=args.data_root, train=True, transform=transforms.ToTensor(), download=True)
            test_dataset = datasets.MNIST(
                root=args.data_root, train=False, transform=transforms.ToTensor())
        elif args.dataset == "SVHN":
            train_dataset = datasets.SVHN(
                root=args.data_root, split="train", transform=transforms.Compose(_SVHN_TRAIN_TRANSFORMS), target_transform=transforms.Lambda(lambda y: y % 10), download=True
            )
            test_dataset = datasets.SVHN(root=args.data_root, split="test", transform=transforms.Compose(_SVHN_TEST_TRANSFORMS),
                                         target_transform=transforms.Lambda(lambda y: y % 10), download=True)
        elif args.dataset == "EMNIST":
            train_dataset = datasets.EMNIST(
                root=args.data_root, split="digits", train=True, transform=transforms.ToTensor(), download=True)
            test_dataset = datasets.MNIST(
                root=args.data_root, split="digits", train=False, transform=transforms.ToTensor(), download=True)
        elif args.dataset == "KMNIST":
            train_dataset = datasets.KMNIST(
                root=args.data_root, train=True, transform=transforms.ToTensor(), download=True)
            test_dataset = datasets.KMNIST(
                root=args.data_root, train=False, transform=transforms.ToTensor(), download=True)
        elif args.dataset == "QMNIST":
            train_dataset = datasets.QMNIST(
                root=args.data_root, what="train", train=True, transform=transforms.ToTensor(), download=True)
            test_dataset = datasets.QMNIST(
                root=args.data_root, what="test", train=False, transform=transforms.ToTensor(), download=True)
        elif args.dataset == "FashionMNIST":
            train_dataset = datasets.FashionMNIST(
                root=args.data_root, train=True, transform=transforms.ToTensor(), download=True)
            test_dataset = datasets.FashionMNIST(
                root=args.data_root, train=False, transform=transforms.ToTensor(), download=True)
        else:
            raise NotImplementedError
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=args.num_workers, pin_memory=True)
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers, pin_memory=True)

        return train_dataloader, test_dataloader
    elif args.exname == "TransferLearning":
        # Load train dataset and test dataset for pretrain and finetune.
        if args.dataset == "MNIST" and args.dataset_aux == "SVHN":
            train_dataset = datasets.MNIST(
                root=args.data_root, train=True, transform=transforms.Compose(_MNIST_COLORIZED_TRAIN_TRANSFORMS), download=True)
            test_dataset = datasets.MNIST(
                root=args.data_root, train=False, transform=transforms.Compose(_MNIST_COLORIZED_TEST_TRANSFORMS), download=True)
            train_dataset_aux = datasets.SVHN(
                root=args.data_root, split="train", transform=transforms.Compose(_SVHN_TRAIN_TRANSFORMS), target_transform=transforms.Lambda(lambda y: y % 10), download=True)
            test_dataset_aux = datasets.SVHN(root=args.data_root, split="test", transform=transforms.Compose(
                _SVHN_TEST_TRANSFORMS), target_transform=transforms.Lambda(lambda y: y % 10), download=True)
        elif args.dataset == "SVHN" and args.dataset_aux == "MNIST":
            train_dataset = datasets.SVHN(
                root=args.data_root, split="train", transform=transforms.Compose(_SVHN_TRAIN_TRANSFORMS), target_transform=transforms.Lambda(lambda y: y % 10), download=True)
            test_dataset = datasets.SVHN(root=args.data_root, split="test", transform=transforms.Compose(
                _SVHN_TEST_TRANSFORMS), target_transform=transforms.Lambda(lambda y: y % 10), download=True)
            train_dataset_aux = datasets.MNIST(
                root=args.data_root, train=True, transform=transforms.Compose(_MNIST_COLORIZED_TRAIN_TRANSFORMS), download=True)
            test_dataset_aux = datasets.MNIST(
                root=args.data_root, train=False, transform=transforms.Compose(_MNIST_COLORIZED_TEST_TRANSFORMS), download=True)
        elif args.dataset == "MNIST" and args.dataset_aux == "QMNIST":
            train_dataset = datasets.MNIST(
                root=args.data_root, train=True, transform=transforms.Compose(_MNIST_COLORIZED_TRAIN_TRANSFORMS), download=True)
            test_dataset = datasets.MNIST(
                root=args.data_root, train=False, transform=transforms.Compose(_MNIST_COLORIZED_TEST_TRANSFORMS), download=True)
            train_dataset_aux = datasets.QMNIST(
                root=args.data_root, what="train", train=True, transform=transforms.ToTensor(), download=True)
            test_dataset_aux = datasets.QMNIST(
                root=args.data_root, what="test", train=False, transform=transforms.ToTensor(), download=True)
        elif args.dataset == "QMNIST" and args.dataset == "MNIST":
            train_dataset = datasets.QMNIST(
                root=args.data_root, what="train", train=True, transform=transforms.ToTensor(), download=True)
            test_dataset = datasets.QMNIST(
                root=args.data_root, what="test", train=False, transform=transforms.ToTensor(), download=True)
            train_dataset_aux = datasets.MNIST(
                root=args.data_root, train=True, transform=transforms.Compose(_MNIST_COLORIZED_TRAIN_TRANSFORMS), download=True)
            test_dataset_aux = datasets.MNIST(
                root=args.data_root, train=False, transform=transforms.Compose(_MNIST_COLORIZED_TEST_TRANSFORMS), download=True)
        else:
            raise NotImplementedError
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=args.num_workers, pin_memory=True)
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers, pin_memory=True)
        train_dataloader_aux = torch.utils.data.DataLoader(
            train_dataset_aux, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=args.num_workers, pin_memory=True)
        test_dataloader_aux = torch.utils.data.DataLoader(
            test_dataset_aux, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers, pin_memory=True)
        return train_dataloader, test_dataloader, train_dataloader_aux, test_dataloader_aux


def get_in_channels(args):
    if args.exname == "TransferLearning":
        return max(_DATASET_CHANNELS[args.dataset], _DATASET_CHANNELS[args.dataset_aux])
    else:
        return _DATASET_CHANNELS[args.dataset]


def get_optimizer(optim_type, lr, net):
    if optim_type == "SGD":
        return optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    elif optim_type == "Adam":
        return optim.Adam(net.parameters(), lr=lr)
    else:
        raise NotImplementedError


def get_model(args):
    afs = AFS if args.af == "all" else [args.af]

    assert "PAU" in afs and not args.cpu or "PAU" not in afs, "PAU need cuda! You can skip the PAU actication functions if you don't have a cuda."
    in_channels = get_in_channels(args)

    model = {af: models.__class_dict__[args.net](
        activations.__class_dict__[af], in_channels) for af in afs}
    model = nn.ModuleDict(model)

    if args.resume is not None:
        model.load_state_dict(torch.load(args.resume), strict=True)
        print("Resume from {}.".format(args.resume))

    model = model if args.cpu else model.cuda()

    return model


class StateKeeper(object):
    def __init__(self, args, state_keeper_name="main"):
        self.args = args
        self.state_keeper_name = state_keeper_name

        os.makedirs("results", exist_ok=True)
        os.makedirs("pretrained", exist_ok=True)

        best_dicts = dict()
        loss_dicts = dict()
        acc_dicts = dict()

        self.model_keys = AFS if args.af == "all" else [args.af]

        for k in self.model_keys:
            best_dicts["first epoch {}".format(k)] = np.zeros(args.times)
            best_dicts["best {}".format(k)] = np.zeros(args.times)
            loss_dicts[k] = [[] for _ in range(args.times)]
            acc_dicts[k] = [[] for _ in range(args.times)]

        self.best_dicts = best_dicts
        self.loss_dicts = loss_dicts
        self.acc_dicts = acc_dicts

    def update(self, time, epoch, loss_dicts, acc_dicts):
        args = self.args

        env_name = "{state_keeper_name}.{prefix}_{time}".format(
            state_keeper_name=self.state_keeper_name, prefix=args.prefix, time=time)

        # VISUALIZE FIRST
        if not args.silent:
            visualize.visualize_losses(
                loss_dicts, title="Loss", env=env_name, epoch=epoch)

            visualize.visualize_accuracy(
                acc_dicts, title="Accuracy", env=env_name, epoch=epoch)

        # STORE
        for k, v in loss_dicts.items():
            self.loss_dicts[k][time].append(v)

        for k, v in acc_dicts.items():
            self.acc_dicts[k][time].append(v)

            if self.best_dicts["first epoch {}".format(k)][time] == 0:
                self.best_dicts["first epoch {}".format(k)][time] = v
                self.best_dicts["best {}".format(k)][time] = v
            else:
                if v > self.best_dicts["best {}".format(k)][time]:
                    self.best_dicts["best {}".format(k)][time] = v

    def save(self):
        args = self.args

        # DRAW CONTINUOUS ERROR BARS
        visualize.ContinuousErrorBars(dicts=self.loss_dicts).draw(
            filename="results/loss.{prefix}.html".format(prefix=args.prefix), ticksuffix="")
        visualize.ContinuousErrorBars(dicts=self.acc_dicts).draw(
            filename="results/acc.{prefix}.html".format(prefix=args.prefix), ticksuffix="%")

        # CALCULATE STATIC
        accuracy = dict()
        for k, v in self.best_dicts.items():
            accuracy["{} mean".format(k)] = np.mean(v)
            accuracy["{} std".format(k)] = np.std(v)
            accuracy["{} best".format(k)] = np.max(v)

        with open("results/{state_keeper_name}.{prefix}.json".format(state_keeper_name=self.state_keeper_name, prefix=args.prefix), "w") as f:
            json.dump(accuracy, f, indent=4)
