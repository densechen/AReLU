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

AFS = list(activations.__class_dict__.keys())
MODELS = list(models.__class_dict__.keys())

parser = argparse.ArgumentParser(description="Activation Player with PyTorch.")
parser.add_argument("--batch_size", default=128, type=int,
                    help="batch size for training")
parser.add_argument("--lr", default=1e-5, type=float, help="learning rate")
parser.add_argument("--epochs", default=20, type=int, help="training epochs")
parser.add_argument("--times", default=5, type=int,
                    help="repeat runing times")
parser.add_argument("--data_root", default="data", type=str,
                    help="the path to MNIST dataset")
parser.add_argument("--num_workers", default=2, type=int,
                    help="number of workers to load data")
parser.add_argument("--net", default="ConvMNIST", choices=MODELS,
                    help="network architecture for experiments. you can add new models in ./models.")
parser.add_argument("--af", default="all", choices=AFS +
                    ["all"], help="the activation function used in experiments. you can specify an activation function by name, or try with all activation functions by `all`")
parser.add_argument("--optim", default="Adam", type=str, choices=["SGD", "Adam"],
                    help="optimizer used in training.")
parser.add_argument("--cuda", action="store_true", default=False,
                    help="with cuda training. this would be much faster.")
parser.add_argument("--exname", default="AFS",
                    help="experiment name of visdom.")
args = parser.parse_args()

args.cuda = True

# 1. BUILD DATASET
train_dataset = datasets.MNIST(
    root=args.data_root, train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(
    root=args.data_root, train=False, transform=transforms.ToTensor())

train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=args.num_workers, pin_memory=True)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers, pin_memory=True)


def define_model_optimizer():
    # 2. DEFINE MODELS
    afs = AFS if args.af == "all" else [args.af]

    assert "PAU" in afs and args.cuda or "PAU" not in afs, "PAU need cuda! You can skip the PAU actication functions if you don't have a cuda."

    model = {af: models.__class_dict__[args.net](
        activations.__class_dict__[af]) for af in afs}
    model = nn.ModuleDict(model)

    model = model.cuda() if args.cuda else model

    # 3. DEFINE OPTIMIZER
    optimizer = optim.SGD(model.parameters(
    ), lr=args.lr, momentum=0.9) if args.optim == "SGD" else optim.Adam(model.parameters(), lr=args.lr)
    return model, optimizer

# 4. TRAIN


def train(model, optimizer):
    model.train()
    process = tqdm(train_dataloader)
    loss_dict = {k: [] for k in model.keys()}
    for data, target in process:
        optimizer.zero_grad()
        data = Variable(data).cuda() if args.cuda else Variable(data)
        target = Variable(target).cuda() if args.cuda else Variable(target)

        for k, v in model.items():
            loss = F.nll_loss(v(data), target)
            loss_dict[k].append(loss.item())
            loss.backward()
        optimizer.step()

    loss_dict = {k: np.mean(v) for k, v in loss_dict.items()}
    return loss_dict

# 5. TEST


def test(model):
    model.eval()
    correct = {k: 0.0 for k in model.keys()}
    process = tqdm(test_dataloader)
    for data, target in process:
        data = Variable(data).cuda() if args.cuda else Variable(data)
        target = Variable(target).cuda() if args.cuda else Variable(target)

        for k, v in model.items():
            pred = v(data).max(1, keepdim=True)[1]
            correct[k] += pred.eq(target.data.view_as(pred)).cpu().sum()

    for k, v in correct.items():
        correct[k] = float(100.0 * v / float(len(test_dataset)))

    return correct


if __name__ == "__main__":
    model, _ = define_model_optimizer()
    testing_accuracy = dict()
    loss_dicts = dict()
    acc_dicts = dict()
    for k in model.keys():
        testing_accuracy["first epoch {}".format(k)] = np.zeros(args.times)
        testing_accuracy["best {}".format(k)] = np.zeros(args.times)
        loss_dicts[k] = [[] for _ in range(args.times)]
        acc_dicts[k] = [[] for _ in range(args.times)]
    del model

    for time in range(args.times):
        model, optimizer = define_model_optimizer()
        for epoch in range(1, args.epochs + 1):
            loss_dict = train(model, optimizer)

            visualize.visualize_losses(loss_dict, title="Loss", env=args.exname +
                                       "-{}-{}_{}".format(args.lr, args.net, time), epoch=epoch)

            for k in loss_dicts.keys():
                loss_dicts[k][time].append(loss_dict[k])

            with torch.no_grad():
                correct = test(model)
                visualize.visualize_accuracy(
                    correct, title="Accuracy", env=args.exname+"-{}-{}_{}".format(args.lr, args.net, time), epoch=epoch)
                # LOG
                for k in acc_dicts.keys():
                    acc_dicts[k][time].append(correct[k])

                for k, v in correct.items():
                    if epoch == 1:
                        testing_accuracy["first epoch {}".format(k)][time] = v
                        testing_accuracy["best {}".format(k)][time] = v
                    else:
                        if v > testing_accuracy["best {}".format(k)][time]:
                            testing_accuracy["best {}".format(k)][time] = v

    # DRAW CONTINUOUS ERROR BARS
    os.makedirs("results", exist_ok=True)
    visualize.ContinuousErrorBars(dicts=loss_dicts).draw(
        filename="results/loss-{}-{}-{}-{}.html".format(args.exname, args.net, args.optim, args.lr), ticksuffix="")
    visualize.ContinuousErrorBars(dicts=acc_dicts).draw(
        filename="results/acc-{}-{}-{}-{}.html".format(args.exname, args.net, args.optim, args.lr), ticksuffix="%")

    # CALCULATE STATIC
    accuracy = dict()
    for k, v in testing_accuracy.items():
        accuracy["{} mean".format(k)] = np.mean(v)
        accuracy["{} std".format(k)] = np.std(v)
        accuracy["{} best".format(k)] = np.max(v)

    with open("results/{}-{}-{}-{}.json".format(args.exname, args.net, args.optim, args.lr), "w") as f:
        json.dump(accuracy, f, indent=4)

    print("Done!")
