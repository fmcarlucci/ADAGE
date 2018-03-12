from enum import Enum

from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np

GAMMA = 0.1

step_down_ratio = 0.4

NESTEROV = True
WEIGHT_DECAY = 1e-4
MOMENTUM = 0.9


class Optimizers(Enum):
    adam = "adam"
    sgd = "sgd"


optimizer_list = [v.value for v in Optimizers]


def get_optimizer_and_scheduler(optim_name, net, max_epochs, lr):
    if optim_name == Optimizers.adam.value:
        return optim.Adam(net.parameters(), lr=lr), None
    elif optim_name == Optimizers.sgd.value:
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY, nesterov=NESTEROV)
        return optimizer, get_scheduler(optimizer, max_epochs)


def get_scheduler(optimizer, max_epochs, step_down=step_down_ratio):
    steps = [max_epochs * k for k in np.arange(step_down, 1.0, step_down)]
    return MultiStepLR(optimizer, milestones=steps, gamma=GAMMA)
