import math

import torch
import torch.nn as nn
from torch.autograd import Function, Variable
from torch.nn import Parameter
from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision.models.alexnet import alexnet
import torch.nn.functional as func


class DecoArgs:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def get_classifier(name, domain_classes, n_classes):
    if name:
        return classifier_list[name](domain_classes, n_classes)
    return CNNModel(domain_classes, n_classes)


def get_net(args):
    if args.use_deco:
        deco_args = DecoArgs(n_layers=args.deco_blocks, train_deco_weight=args.train_deco_weight,
                             deco_kernels=args.deco_kernels, deco_block=deco_types[args.deco_block_type],
                             out_channels=args.deco_output_channels)
        my_net = Combo(deco_args, classifier=args.classifier, domain_classes=args.domain_classes,
                       n_classes=args.n_classes)
    else:
        my_net = get_classifier(args.classifier, domain_classes=args.domain_classes, n_classes=args.n_classes)

    for p in my_net.parameters():
        p.requires_grad = True
    return my_net


def entropy_loss(x):
    return torch.sum(-func.softmax(x, 1) * func.log_softmax(x, 1), 1).mean()


deco_types = {'basic': BasicBlock, 'bottleneck': Bottleneck}


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
    def __init__(self, deco_args, classifier, domain_classes=2, n_classes=10):
        super(Combo, self).__init__()
        self.net = get_classifier(classifier, domain_classes, n_classes)
        if isinstance(self.net, AlexNet):
            self.deco = DECO(deco_args)
        else:
            self.deco = DECO_mini(deco_args)

    def forward(self, input_data, lambda_val):
        input_data = self.deco(input_data)
        return self.net(input_data, lambda_val)


class BasicDECO(nn.Module):
    def __init__(self, deco_args):
        super(BasicDECO, self).__init__()
        self.inplanes = deco_args.deco_kernels
        self.ratio = 1.0
        if deco_args.train_deco_weight:
            self.deco_weight = Parameter(torch.FloatTensor(1), requires_grad=True)
        else:
            self.deco_weight = Variable(torch.FloatTensor(1)).cuda()
        if deco_args.train_image_weight:
            self.image_weight = Parameter(torch.FloatTensor(1), requires_grad=True)
        else:
            self.image_weight = Variable(torch.FloatTensor(1)).cuda()
        self.deco_weight.data.fill_(deco_args.deco_weight)
        self.image_weight.data.fill_(1.0)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def weighted_sum(self, input_data, x):
        x = self.deco_weight * x
        input_data = self.image_weight * input_data
        self.ratio = input_data.norm() / x.norm()
        return x + input_data

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


class DECO_mini(BasicDECO):
    def __init__(self, deco_args):
        super(DECO_mini, self).__init__(deco_args)
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=5, stride=1, padding=2,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(deco_args.block, self.inplanes, deco_args.n_layers)
        self.conv_out = nn.Conv2d(self.inplanes * deco_args.block.expansion, deco_args.output_channels, 1)
        self.init_weights()
        print(self)

    def forward(self, input_data):
        input_data = input_data.expand(input_data.data.shape[0], 3, input_data.data.shape[2], input_data.data.shape[3])
        x = self.conv1(input_data)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.conv_out(x)

        return self.weighted_sum(input_data, x)


class DECO(BasicDECO):
    def __init__(self, deco_args):
        super(DECO, self).__init__(deco_args)
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=5, stride=2, padding=2,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(deco_args.block, self.inplanes, deco_args.n_layers)
        self.conv_out = nn.Conv2d(self.inplanes * deco_args.block.expansion, deco_args.output_channels, 1)
        self.init_weights()
        print(self)

    def forward(self, input_data):
        input_data = input_data.expand(input_data.data.shape[0], 3, input_data.data.shape[2], input_data.data.shape[3])
        x = self.conv1(input_data)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.conv_out(x)
        x = nn.functional.upsample(x, scale_factor=4, mode='bilinear')

        return self.weighted_sum(input_data, x)


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
    def __init__(self, domain_classes, n_classes):
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
            nn.Linear(100, domain_classes)
        )
        self.class_classifier = nn.Sequential(
            nn.Linear(48 * 4 * 4, 100),
            nn.ReLU(True),
            nn.Linear(100, 100),
            nn.ReLU(True),
            nn.Linear(100, n_classes)
        )


class SVHNModel(BasicDANN):
    def __init__(self, domain_classes, n_classes):
        super(SVHNModel, self).__init__()
        print("Using SVHN")
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 5, padding=2),
            nn.Dropout(0.1, True),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2, padding=1),
            nn.Conv2d(64, 64, 5, padding=2),
            nn.Dropout(0.25, True),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2, padding=1),
            nn.Conv2d(64, 128, 5, padding=2),
            nn.Dropout(0.25, True),
            nn.ReLU(True)
        )
        self.domain_classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 1024),
            nn.Dropout(0.5, True),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.Dropout(0.5, True),
            nn.ReLU(True),
            nn.Linear(1024, domain_classes)
        )
        self.class_classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 3072),
            nn.Dropout(0.5, True),
            nn.ReLU(True),
            nn.Linear(3072, 2048),
            nn.Dropout(0.5, True),
            nn.ReLU(True),
            nn.Linear(2048, n_classes)
        )


class CNNModel(nn.Module):
    def __init__(self, domain_classes, n_classes):
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
        self.class_classifier.add_module('c_fc3', nn.Linear(100, n_classes))
        # self.class_classifier.add_module('c_softmax', nn.LogSoftmax(1))

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(50 * 4 * 4, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm2d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, domain_classes))

    def forward(self, input_data, lambda_val):
        input_data = input_data.expand(input_data.data.shape[0], 3, 28, 28)
        feature = self.feature(input_data)
        feature = feature.view(-1, 50 * 4 * 4)
        reverse_feature = ReverseLayerF.apply(feature, lambda_val)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output


class AlexNet(BasicDANN):
    def __init__(self, domain_classes, n_classes):
        super(AlexNet, self).__init__()
        pretrained = alexnet()
        self._convs = pretrained.features
        self.bottleneck = nn.Linear(4096, 256),  # bottleneck
        self._classifier = nn.Sequential(
            nn.Dropout(),
            pretrained.classifier[1],  # nn.Linear(256 * 6 * 6, 4096),  #
            nn.ReLU(inplace=True),
            nn.Dropout(),
            pretrained.classifier[4],  # nn.Linear(4096, 4096),  #
            nn.ReLU(inplace=True),
            self.bottleneck
        )
        self.features = nn.Sequential(self._convs, self._classifier)
        self.class_classifier = nn.Linear(256, n_classes)
        self.domain_classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256, 1024),  # pretrained.classifier[1]
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 1024),  # pretrained.classifier[4]
            nn.ReLU(inplace=True),
            nn.Linear(1024, domain_classes),
        )


classifier_list = {"roided_lenet": CNNModel,
                   "mnist": MnistModel,
                   "svhn": SVHNModel,
                   "alexnet": AlexNet}
