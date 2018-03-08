import math

import torch
import torch.nn as nn
from torch.autograd import Function, Variable
from torch.nn import Parameter
from torchvision.models.resnet import BasicBlock


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, lambda_val):
        ctx.lambda_val = lambda_val

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.lambda_val

        return output, None


class Combo(nn.Module):
    def __init__(self, deco_weight=0.001, n_deco=4, deco_block=BasicBlock, classifier=None, train_deco_weight=False):
        super(Combo, self).__init__()
        self.deco = Deco(deco_block, [n_deco], deco_weight, train_deco_weight)
        self.net = get_classifier(classifier)

    def forward(self, input_data, lambda_val):
        input_data = self.deco(input_data)
        return self.net(input_data, lambda_val)


class Deco(nn.Module):
    def __init__(self, block, layers, deco_weight, train_deco_weight):
        self.inplanes = 64
        self.ratio = 1
        super(Deco, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        #        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.conv3D = nn.Conv2d(64, 3, 1)
        # self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        if train_deco_weight:
            self.deco_weight = Parameter(torch.FloatTensor(1), requires_grad=True)
        else:
            self.deco_weight = Variable(torch.FloatTensor(1)).cuda()
        self.deco_weight.data.fill_(deco_weight)

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

    def forward(self, input_data):
        input_data = input_data.expand(input_data.data.shape[0], 3, 28, 28)
        x = self.conv1(input_data)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        x = self.layer1(x)
        x = self.conv3D(x)
        #        x = self.layer2(x)
        # x = nn.functional.upsample(x, scale_factor=2, mode='bilinear')
        x = self.deco_weight * x
        self.ratio = input_data.norm() / x.norm()
        return input_data + x  # , x.norm() / input_data.shape[0]


class BasicDANN(nn.Module):
    def __init__(self):
        super(BasicDANN, self).__init__()
        self.features = None
        self.domain_classifier = None
        self.class_classifier = None

    def forward(self, input_data, lambda_val):
        input_data = input_data.expand(input_data.data.shape[0], 3, input_data.data.shape[2], input_data.data.shape[3])
        feature = self.features(input_data)
        feature = feature.view(input_data.shape[0], -1)
        reverse_feature = ReverseLayerF.apply(feature, lambda_val)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output


class MnistModel(BasicDANN):
    def __init__(self):
        super(MnistModel, self).__init__()
        print("Using LeNet")
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 5),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 48, 5),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        self.domain_classifier = nn.Sequential(
            nn.Linear(48 * 4 * 4, 100),
            nn.ReLU(True),
            nn.Linear(100, 2),
            nn.LogSoftmax(1)
        )
        self.class_classifier = nn.Sequential(
            nn.Linear(48 * 4 * 4, 100),
            nn.ReLU(True),
            nn.Linear(100, 100),
            nn.ReLU(True),
            nn.Linear(100, 10),
            nn.LogSoftmax(1)
        )


class SVHNModel(BasicDANN):
    def __init__(self):
        super(SVHNModel, self).__init__()
        print("Using SVHN")
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 5, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2, padding=1),
            nn.Conv2d(64, 64, 5, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2, padding=1),
            nn.Conv2d(64, 128, 5, padding=2),
            nn.ReLU(True)
        )
        self.domain_classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 2),
            nn.LogSoftmax(1)
        )
        self.class_classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 3072),
            nn.ReLU(True),
            nn.Linear(3072, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 10),
            nn.LogSoftmax(1)
        )


def get_classifier(name):
    if name:
        return classifier_list[name]()
    return CNNModel()


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.feature = nn.Sequential()
        self.feature.add_module('f_conv1', nn.Conv2d(3, 64, kernel_size=5))
        self.feature.add_module('f_bn1', nn.BatchNorm2d(64))
        self.feature.add_module('f_pool1', nn.MaxPool2d(2))
        self.feature.add_module('f_relu1', nn.ReLU(True))
        self.feature.add_module('f_conv2', nn.Conv2d(64, 50, kernel_size=5))
        self.feature.add_module('f_bn2', nn.BatchNorm2d(50))
        self.feature.add_module('f_drop1', nn.Dropout2d())
        self.feature.add_module('f_pool2', nn.MaxPool2d(2))
        self.feature.add_module('f_relu2', nn.ReLU(True))

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(50 * 4 * 4, 100))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm2d(100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout2d())
        self.class_classifier.add_module('c_fc2', nn.Linear(100, 100))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm2d(100))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(100, 10))
        self.class_classifier.add_module('c_softmax', nn.LogSoftmax(1))

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(50 * 4 * 4, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm2d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(1))

    def forward(self, input_data, lambda_val):
        input_data = input_data.expand(input_data.data.shape[0], 3, 28, 28)
        feature = self.feature(input_data)
        feature = feature.view(-1, 50 * 4 * 4)
        reverse_feature = ReverseLayerF.apply(feature, lambda_val)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output


classifier_list = {"roided_lenet": CNNModel,
                   "mnist": MnistModel,
                   "svhn": SVHNModel}
