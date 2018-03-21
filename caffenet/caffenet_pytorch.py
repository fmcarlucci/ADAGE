import torch
import torch.nn as nn

from models.torch_future import LocalResponseNorm, Flatten

def load_caffenet():
    model = nn.Sequential(  # Sequential,
        nn.Conv2d(3, 96, (11, 11), (4, 4)), #0
        nn.ReLU(),
        nn.MaxPool2d((3, 3), (2, 2), (0, 0), ceil_mode=True),
        LocalResponseNorm(5, 0.0001, 0.75),
        nn.Conv2d(96, 256, (5, 5), (1, 1), (2, 2), 1, 2), #4
        nn.ReLU(),
        nn.MaxPool2d((3, 3), (2, 2), (0, 0), ceil_mode=True),
        LocalResponseNorm(5, 0.0001, 0.75),
        nn.Conv2d(256, 384, (3, 3), (1, 1), (1, 1)), #8
        nn.ReLU(),
        nn.Conv2d(384, 384, (3, 3), (1, 1), (1, 1), 1, 2), #10
        nn.ReLU(),
        nn.Conv2d(384, 256, (3, 3), (1, 1), (1, 1), 1, 2), #12
        nn.ReLU(),
        nn.MaxPool2d((3, 3), (2, 2), (0, 0), ceil_mode=True),
        Flatten(),
        nn.Linear(9216, 4096),  # Linear, 16
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096, 4096),  # Linear, 19
        nn.ReLU(),
        nn.Dropout(0.5),
    )
    pretrained = torch.load('caffenet/caffenet_pytorch.pth')
    pretrained["16.weight"] = pretrained["16.1.weight"]
    pretrained["16.bias"] = pretrained["16.1.bias"]
    pretrained["19.weight"] = pretrained["19.1.weight"]
    pretrained["19.bias"] = pretrained["19.1.bias"]
    del pretrained["16.1.weight"], pretrained["16.1.bias"], pretrained["19.1.weight"], pretrained["19.1.bias"], \
    pretrained["22.1.weight"], pretrained["22.1.bias"]
    model.load_state_dict(pretrained)
    return  model