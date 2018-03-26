from enum import Enum

from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np

GAMMA = 0.1

base_step_down_ratio = 0.4

NESTEROV = True
WEIGHT_DECAY = 0
MOMENTUM = 0.9


class Optimizers(Enum):
    adam = "adam"
    sgd = "sgd"


optimizer_list = [v.value for v in Optimizers]


def get_optimizer_and_scheduler(optim_name, net, max_epochs, lr, keep_pretrained_fixed):
    if keep_pretrained_fixed:
        params = net.get_trainable_params
    else:
        params = net.parameters
    print("Number of trainable group of params %d:" % sum(1 for x in params()))
    if optim_name == Optimizers.adam.value:
        optimizer = optim.Adam(params(), lr=lr)
        step_down_ratio = 0.8
    elif optim_name == Optimizers.sgd.value:
        optimizer = optim.SGD(params(), lr=lr, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY, nesterov=NESTEROV)
        step_down_ratio = base_step_down_ratio
    return optimizer, get_scheduler(optimizer, max_epochs, step_down_ratio)


def get_scheduler(optimizer, max_epochs, step_down=base_step_down_ratio):
    steps = [max_epochs * k for k in np.arange(step_down, 1.0, step_down)]
    return MultiStepLR(optimizer, milestones=steps, gamma=GAMMA)
