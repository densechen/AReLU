# AReLU: Attention-based-Rectified-Linear-Unit

Activation Function Player with PyTorch.

## 1. Introduction

This repository is the implementation of paper [AReLU: Attention-based-Rectified-Linear-Unit](AReLU). 

While developing, we found that this repo is quiet convenient for people to do different kind of experiments with different activation functions, different learning rating, different optimizer and different network structure. And it is easy for us to add new activation functions and new network structures into program. What's more, based on visdom and ploty, we also provide a quite nice visualization of training process and training results.

This project is friendly to newcomers of PyTorch.

## 2. Install

```shell
conda create -n AFP python=3.7 -y
conda activate AFP
pip install -r requirements.txt
```

**NOTE**: PAU is only CUDA supported. You have to compile it first:
``` shell
pip install airspeed==0.5.14 

cd activations/pau/cuda
python setup.py install
```

The code of PAU is directly token from [PAU](https://github.com/ml-research/pau.git), if you occur any problems while compiling, please refer to the original repository.

If you just want to have a quick start, and do not want to compile with PAU, just comment out the following lines in [activations/\_\_init\_\_.py](activations/__init__.py):

```python
try:
    from .pau.utils import PAU
    __class_dict__["PAU"] = PAU
except Exception:
    raise NotImplementedError("")
```

## 3. Run

### Prepare

We are using visdom to visualize the training process. 
Before training, please setup the visdom server:
```shell
python -m visdom.server &
```
Now, you can refer to "http://localhost:8097/" for more training information.

**NOTE**: Don't worry about the training data. The program will download the MNIST dataset while runtime and save it under `args.data_root`

### Quick start
If you want to have a quick start with default parameters, just run:
```shell
python main.py --cuda
```

We will plot the Continuous Error Bars with ploty and save it as a html file under `results` folder. A json file which records same static data is also generated and saved under `results`.

Training loss (visualzie on visdom: http://localhost:8097/):

![loss](pictures/loss.png)

Testing accuracy (visualize on visdom: http://localhost:8097/):

![acc](pictures/acc.png)

Continuous Error Bars of training loss with five runs (saved under `results` as html file):

![loss_ceb](pictures/loss_ceb.png)

Continuous Error Bars of testing accuracy with five runs (saved under `results` as html file):

![acc_ceb](pictures/acc_ceb.png)

### Run with different parameters
You can try with more flexible parameters by:

```shell
python main.py -h
    usage: main.py [-h] [--batch_size BATCH_SIZE] [--lr LR] [--epochs EPOCHS]
                [--times TIMES] [--data_root DATA_ROOT]
                [--num_workers NUM_WORKERS]
                [--net {BaseModel,ConvMNIST,LinearMNIST}]
                [--af {APL,AReLU,GELU,Maxout,Mixture,SLAF,Swish,ReLU,ReLU6,Sigmoid,LeakyReLU,ELU,PReLU,SELU,Tanh,RReLU,CELU,Softplus,PAU,all}]
                [--optim {SGD,Adam}] [--cuda] [--exname EXNAME]

    Activation Player with PyTorch.

    optional arguments:
    -h, --help            show this help message and exit
    --batch_size BATCH_SIZE
                            batch size for training
    --lr LR               learning rate
    --epochs EPOCHS       training epochs
    --times TIMES         repeat runing times
    --data_root DATA_ROOT
                            the path to MNIST dataset
    --num_workers NUM_WORKERS
                            number of workers to load data
    --net {BaseModel,ConvMNIST,LinearMNIST}
                            network architecture for experiments. you can add new
                            models in ./models.
    --af {APL,AReLU,GELU,Maxout,Mixture,SLAF,Swish,ReLU,ReLU6,Sigmoid,LeakyReLU,ELU,PReLU,SELU,Tanh,RReLU,CELU,Softplus,PAU,all}
                            the activation function used in experiments. you can
                            specify an activation function by name, or try with
                            all activation functions by `all`
    --optim {SGD,Adam}    optimizer used in training.
    --cuda                with cuda training. this would be much faster.
    --exname EXNAME       experiment name of visdom.
```

### Do a full training
We provide a script for doing a full training with different activation functions, learning rates, optimizers and network structure.

Just run:
```shell
./train.sh
```

**NOTE**: This step is time consuming.

## 4. Explore

### New activation functions
1. write a python script file under `activations`, such as *new_activation_functions.py*, which contains the implementation of new activation functions. 
2. import new activation functions in [activations/\_\_init\_\_.py](activations/__init__.py), like:

```python
from .new_activation_functions import NewActivationFunctions
```
3. Enjoy it! 

### New network structure
1. Write a python script file under `models`, such as *new_network_structure.py*, which contains the definition of new network structure. New defined network structure should be a subclass of **BaseModel**, which defined in `models/models.py`. Such as:

```python
from models import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearMNIST(BaseModel):
    def __init__(self, activation: nn.Module):
        super().__init__(activation)

        self.linear1 = nn.Sequential(
            nn.Linear(28 * 28, 512),
            activation(),
        )

        self.linear2 = nn.Sequential(
            nn.Linear(512, 10),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x):
        x = x.view(-1, 28 * 28)

        x = self.linear1(x)

        x = self.linear2(x)

        return x
```

2. Import new network structure in [models/\_\_init\_\_/py](models/__init__.py), like:
```python
from .linear import LinearMNIST
```

3. Enjoy it!

### More
You can modify `main.py` to try with more datasets and optimizer.

## 5. More tasks

### Classification

You can refer to [CIFAR10](https://github.com/kuangliu/pytorch-cifar.git) and [CIFAR100](https://github.com/weiaicunzai/pytorch-cifar100.git) for more experiments with popular network structure. 
After download the repo, you just copy `activations` folder into repo, and modify some code.

### Segmentation 

You can refer to [Detectron2](https://github.com/facebookresearch/detectron2.git) for more experiments on segmentation. And refer to [UNet-Brain](https://github.com/mateuszbuda/brain-segmentation-pytorch.git) for a simple test with UNet on brain segmentation.