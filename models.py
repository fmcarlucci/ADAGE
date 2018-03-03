import math
from itertools import chain

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
from torchvision.models.resnet import Bottleneck

# code adapted from https://github.com/SSARCandy/DeepCORAL
import old_models


def get_new_image(input, model):
    return model.sharedNet[0](input)


deco_weight = 0.001


class DecoReverseGrade(nn.Module):
    def __init__(self, num_classes=1000, deco_weight=deco_weight):
        super(DecoReverseGrade, self).__init__()
        self.deco = Deco(Bottleneck, [8], deco_weight)


class DeepCORAL(nn.Module):
    def __init__(self, num_classes=1000):
        super(DeepCORAL, self).__init__()
        self.sharedNet = AlexNet()
        self.source_fc = nn.Linear(4096, num_classes)
        self.target_fc = nn.Linear(4096, num_classes)

        # initialize according to CORAL paper experiment
        self.source_fc.weight.data.normal_(0, 0.005)
        self.target_fc.weight.data.normal_(0, 0.005)
        self.deco = self.sharedNet
        for param in self.sharedNet.parameters():
            param.requires_grad = True

    def forward(self, source, target):
        source = self.sharedNet(source)
        target = self.sharedNet(target)
        self.fc7coral = old_models.CORAL(source, target)

        source = self.source_fc(source)
        target = self.source_fc(target)
        return source, target, Variable(torch.zeros(1)).cuda()

    def get_fc7_coral(self):
        return self.fc7coral


class DeepColorizationCORAL(nn.Module):
    def __init__(self, num_classes=1000, deco_weight=deco_weight):
        super(DeepColorizationCORAL, self).__init__()
        self.deco = Deco(Bottleneck, [8], deco_weight)
        self.sharedNet = AlexNet()
        self.source_fc = nn.Linear(4096, num_classes)
        self.target_fc = nn.Linear(4096, num_classes)

        # initialize according to CORAL paper experiment
        self.source_fc.weight.data.normal_(0, 0.005)
        self.target_fc.weight.data.normal_(0, 0.005)
        self.fc7coral = None
        self.fc6coral = None

    def forward(self, source, target):
        source, source_res_norm = self.deco(source)
        source = self.sharedNet(source)

        target, target_res_norm = self.deco(target)
        target = self.sharedNet(target)

        self.fc7coral = old_models.CORAL(source, target)
        source = self.source_fc(source)
        target = self.source_fc(target)
        return source, target, source_res_norm + target_res_norm

    def get_fc7_coral(self):
        return 1000 * self.fc7coral


class DeepColorizationCORAL_targetOnly(DeepColorizationCORAL):
    def __init__(self, num_classes=1000, deco_weight=deco_weight):
        super(DeepColorizationCORAL_targetOnly, self).__init__()

    def forward(self, source, target):
        source = self.sharedNet(source)

        target, target_res_norm = self.deco(target)
        target = self.sharedNet(target)
        source_res_norm = target_res_norm

        self.fc7coral = old_models.CORAL(source, target)
        source = self.source_fc(source)
        target = self.source_fc(target)
        return source, target, source_res_norm + target_res_norm


class DeepColorizationCORAL_sourceOnly(DeepColorizationCORAL):
    def __init__(self, num_classes=1000, deco_weight=deco_weight):
        super(DeepColorizationCORAL_sourceOnly, self).__init__()

    def forward(self, source, target):
        # source, source_res_norm = self.deco(source)
        source, source_res_norm = self.deco(source)
        source = self.sharedNet(source)

        target_res_norm = source_res_norm
        target = self.sharedNet(target)

        self.fc7coral = old_models.CORAL(source, target)
        source = self.source_fc(source)
        target = self.source_fc(target)
        return source, target, source_res_norm + target_res_norm


class Deco(nn.Module):
    def __init__(self, block, layers, deco_weight):
        self.inplanes = 64
        super(Deco, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        #        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.conv3D = nn.Conv2d(256, 3, 1)
        # self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        self.deco_weight = deco_weight

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, original):
        x = self.conv1(original)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.conv3D(x)
        #        x = self.layer2(x)
        x = self.deco_weight * nn.functional.upsample(x, scale_factor=4, mode='bilinear')
        return original + x, x.norm() / original.shape[0]


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            # nn.Linear(4096, num_classes),
        )
        for param in chain(self.features.parameters(), self.classifier.parameters()):
            param.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


def load_pretrained(model):
    url = 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth'
    pretrained_dict = model_zoo.load_url(url)
    model_dict = model.state_dict()

    # filter out unmatch dict and delete last fc bias, weight
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # del pretrained_dict['classifier.6.bias']
    # del pretrained_dict['classifier.6.weight']

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
