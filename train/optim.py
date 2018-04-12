from enum import Enum

import numpy as np
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR, _LRScheduler

GAMMA = 0.1

base_step_down_ratio = 0.4

NESTEROV = True
WEIGHT_DECAY = 0.0005
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
    scheduler = get_scheduler(optimizer, max_epochs, step_down_ratio)
    return optimizer, scheduler


def get_scheduler(optimizer, max_epochs, step_down=base_step_down_ratio):
    steps = [max_epochs * k for k in np.arange(step_down, 1.0, step_down)]
    return MultiStepLR(optimizer, milestones=steps, gamma=GAMMA)


class InvertedLR(_LRScheduler):
    """Set the learning rate of each parameter group to the initial lr decayed
    by gamma every epoch. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        gamma (float): Multiplicative factor of learning rate decay.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, gamma, power, init_lr, last_epoch=-1):
        self.gamma = gamma
        self.power = power
        self.init_lr = init_lr
        self.iter = -1
        super(InvertedLR, self).__init__(optimizer, last_epoch)

    def step_iter(self):
        self.iter += 1
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    def get_lr(self):
        return [base_lr * (self.init_lr * (1 + self.gamma * self.iter) ** (-self.power))
                for base_lr in self.base_lrs]

    def step(self, epoch=None):
        pass