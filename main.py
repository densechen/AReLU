import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from tqdm import tqdm

import activations
import models
import visualize
import utils

AFS = list(activations.__class_dict__.keys())
MODELS = list(models.__class_dict__.keys())

parser = argparse.ArgumentParser(
    description="Activation Function Player with PyTorch.")
parser.add_argument("--batch_size", default=128, type=int,
                    help="batch size for training")
parser.add_argument("--lr", default=1e-5, type=float, help="learning rate")
parser.add_argument("--lr_aux", default=1e-5, type=float,
                    help="learning rate of finetune. only used while transfer learning.")
parser.add_argument("--epochs", default=2, type=int, help="training epochs")
parser.add_argument("--epochs_aux", default=2, type=int,
                    help="training epochs. only used while transfer learning.")
parser.add_argument("--times", default=2, type=int,
                    help="repeat runing times")
parser.add_argument("--data_root", default="data", type=str,
                    help="the path to dataset")
parser.add_argument("--dataset", default="MNIST",
                    choices=utils._DATASET_CHANNELS.keys(), help="the dataset to play with.")
parser.add_argument("--dataset_aux", default="SVHN", choices=utils._DATASET_CHANNELS.keys(),
                    help="the dataset to play with. only used while transfer learning.")
parser.add_argument("--num_workers", default=2, type=int,
                    help="number of workers to load data")
parser.add_argument("--net", default="ConvMNIST", choices=MODELS,
                    help="network architecture for experiments. you can add new models in ./models.")
parser.add_argument("--resume", default=None, help="pretrained path to resume")
parser.add_argument("--af", default="all", choices=AFS +
                    ["all"], help="the activation function used in experiments. you can specify an activation function by name, or try with all activation functions by `all`")
parser.add_argument("--optim", default="SGD", type=str, choices=["SGD", "Adam"],
                    help="optimizer used in training.")
parser.add_argument("--cpu", action="store_true", default=False,
                    help="with cuda training. this would be much faster.")
parser.add_argument("--exname", default="AFS", choices=["AFS", "TransferLearning"],
                    help="experiment name of visdom.")
parser.add_argument("--silent", action="store_true", default=False,
                    help="if True, shut down the visdom visualizer.")
args = parser.parse_args()
args.prefix = "{exname}.{dataset}.{dataset_aux}.{net}.{af}.{optim}.{lr}.{lr_aux}.{epochs}.{epochs_aux}.{batch_size}".format(
    exname=args.exname, dataset=args.dataset, dataset_aux=args.dataset_aux, net=args.net, af=args.af, optim=args.optim,
    lr=args.lr, lr_aux=args.lr_aux, epochs=args.epochs, epochs_aux=args.epochs_aux, batch_size=args.batch_size
)

# 1. BUILD DATASET
if args.exname == "AFS":
    train_dataloader, test_dataloader = utils.get_loader(args)
elif args.exname == "TransferLearning":
    train_dataloader, test_dataloader, train_dataloader_aux, test_dataloader_aux = utils.get_loader(
        args)
else:
    raise ValueError

# 4. TRAIN


def train(model, optimizer, dataloader):
    model.train()
    process = tqdm(dataloader)
    loss_dict = {k: [] for k in model.keys()}
    for data, target in process:
        optimizer.zero_grad()
        data = Variable(data).cuda() if not args.cpu else Variable(data)
        target = Variable(target).cuda() if not args.cpu else Variable(target)

        for k, v in model.items():
            loss = F.nll_loss(v(data), target)
            loss_dict[k].append(loss.item())
            loss.backward()
        optimizer.step()

    loss_dict = {k: np.mean(v) for k, v in loss_dict.items()}
    return loss_dict

# 5. TEST


def test(model, dataloader):
    model.eval()
    correct = {k: 0.0 for k in model.keys()}
    process = tqdm(dataloader)
    for data, target in process:
        data = Variable(data).cuda() if not args.cpu else Variable(data)
        target = Variable(target).cuda() if not args.cpu else Variable(target)

        for k, v in model.items():
            pred = v(data).max(1, keepdim=True)[1]
            correct[k] += pred.eq(target.data.view_as(pred)).cpu().sum()

    for k, v in correct.items():
        correct[k] = float(100.0 * v / len(dataloader.dataset))

    return correct


def forward_epoch(model, train_dataloader, test_dataloader, optimizer, state_keeper, time, epochs):
    for epoch in range(1, epochs + 1):
        loss_dict = train(model, optimizer, train_dataloader)
        with torch.no_grad():
            correct = test(model, test_dataloader)

        state_keeper.update(time, epoch, loss_dict, correct)

        save_path = "pretrained/{prefix}.{time}.pth".format(
            prefix=args.prefix, time=time)
        torch.save(model.state_dict(), f=save_path)
        print("Current model has been saved under {}.".format(save_path))


if __name__ == "__main__":
    state_keeper = utils.StateKeeper(args)
    if args.exname == "TransferLearning":
        state_keeper_aux = utils.StateKeeper(args, state_keeper_name="aux")

    for time in range(args.times):
        model = utils.get_model(args)
        optimizer = utils.get_optimizer(args.optim, args.lr, model)
        forward_epoch(model, train_dataloader, test_dataloader,
                      optimizer, state_keeper, time, args.epochs)
        if args.exname == "TransferLearning":
            optimizer_aux = utils.get_optimizer(
                args.optim, args.lr_aux, model)
            forward_epoch(model, train_dataloader_aux, test_dataloader_aux, optimizer_aux, state_keeper_aux,
                          time, args.epochs_aux)

    state_keeper.save()
    if args.exname == "TransferLearning":
        state_keeper_aux.save()
    print("Done!")
