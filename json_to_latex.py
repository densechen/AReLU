import json
from glob import glob

import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Json to LaTex (Lightly)")
parser.add_argument("--exname", default="AFP", help="exname to generate json")
parser.add_argument("--data", default="best", choices=[
                    "best", "mean", "std"], help="best: best accuracy, mean: mean accuracy, std: std of acc")
parser.add_argument("--epoch", default="first epoch",
                    choices=["first epoch", "best"], help="which epoch to load.")
parser.add_argument("--output", default="latex.txt", help="output filename")
args = parser.parse_args()

if __name__ == "__main__":
    # 1. LOAD ALL JSON FILE
    json_file_list = glob("results/{}*.json".format(args.exname))

    json_file = {}
    for fn in json_file_list:
        with open(fn, "r") as fp:
            key = fn.split("/")[1]
            json_file[key] = json.load(fp)

    # 2. GENERATE TABLE
    optimizer = []
    learning_rate = []
    network = []
    activation = []

    for k, v in json_file.items():
        k = str(k)
        k = k.replace("1e-05", "0.00001")
        ks = k.split("-")
        optimizer.append(ks[2])
        network.append(ks[1])
        learning_rate.append(ks[3].replace(".json", ""))

        for kk, vv in v.items():
            kk = str(kk)
            if kk.startswith("best"):
                activation.append(kk.split(" ")[1])

    optimizer = sorted(set(optimizer))
    learning_rate = sorted(set(learning_rate))
    learning_rate.reverse()
    network = sorted(set(network))
    activation = sorted(set(activation))

    # 3. CREATE LATEX CODE
    # first row: learning rate
    # second row: Net. Optim
    # total col: 1 + num_leanring_rate \times num_net \times num_optim
    total_col = 1 + len(learning_rate) * len(network) * len(optimizer)

    latex = open(args.output, "w")
    # 1. first row
    line = [[lr] * len(network) * len(optimizer) for lr in learning_rate]
    line = np.array(line).reshape(-1)
    latex.write("Learing Rate & " + "&".join(line) + '\\\\ \n')
    # 2. second row
    latex.write("Net. Optim & " + "&".join(["{}. {}".format(net, optim)
                                            for net in network for optim in optimizer] * len(learning_rate)) + "\\\\ \n")

    # 3. write data
    for act in activation:
        line = "{}".format(act)

        for lr in learning_rate:
            if lr == "0.00001":
                lr = "1e-05"
            for net in network:
                for optim in optimizer:
                    jf = "{}-{}-{}.json".format(net, optim, lr)
                    for k, v in json_file.items():
                        if jf in k:
                            it = "{} {} {}".format(args.epoch, act, args.data)
                            line += "& {:.2f}".format(v[it])
                            break
        latex.write(line + "\\\\ \n")

    latex.close()
