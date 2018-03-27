import math

import itertools
import torch
import torch.nn as nn
from torch.autograd import Function, Variable
from torch.nn import Parameter
from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision.models.alexnet import alexnet
import torch.nn.functional as func

from caffenet.caffenet_pytorch import load_caffenet
from models.torch_future import Flatten

deco_starting_weight = 0.0001


class DecoArgs:
    def __init__(self, args):
        self.n_layers = args.deco_blocks
        self.train_deco_weight = args.train_deco_weight
        self.train_image_weight = args.train_image_weight
        self.deco_kernels = args.deco_kernels
        self.block = deco_types[args.deco_block_type]
        self.output_channels = args.deco_output_channels
        self.deco_weight = deco_starting_weight
        self.no_residual = args.deco_no_residual
        self.mode = args.deco_mode
        self.use_tanh = args.deco_tanh
        self.no_pool = args.deco_no_pool
        self.deconv = args.deco_deconv


def get_classifier(name, domain_classes, n_classes):
    if name:
        return classifier_list[name](domain_classes, n_classes)
    return CNNModel(domain_classes, n_classes)


def get_net(args):
    if args.use_deco:
        deco_args = DecoArgs(args)
        my_net = deco_modes[deco_args.mode](deco_args, classifier=args.classifier, domain_classes=args.domain_classes,
                                            n_classes=args.n_classes)
    else:
        my_net = get_classifier(args.classifier, domain_classes=args.domain_classes, n_classes=args.n_classes)

    for p in my_net.parameters():
        p.requires_grad = True
    print(my_net)
    return my_net


def entropy_loss(x):
    return torch.sum(-func.softmax(x, 1) * func.log_softmax(x, 1), 1).mean()


deco_types = {'basic': BasicBlock, 'bottleneck': Bottleneck}


# Utility class for the combo network
class PassData(nn.Module):
    def forward(self, input_data):
        return input_data


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
        if isinstance(self.net, AlexNetStyleDANN):
            self.deco_architecture = DECO
        else:
            self.deco_architecture = DECO_mini

    def set_deco_mode(self, mode):
        self.deco = self.domain_transforms[mode]

    def forward(self, input_data, lambda_val):
        input_data = self.deco(input_data)
        return self.net(input_data, lambda_val)

    def get_trainable_params(self):
        return itertools.chain(self.get_deco_parameters(), self.net.get_trainable_params())


class SourceOnlyCombo(Combo):
    def __init__(self, deco_args, classifier, domain_classes=2, n_classes=10):
        super(SourceOnlyCombo, self).__init__(deco_args, classifier, domain_classes, n_classes)
        self.source = self.deco_architecture(deco_args)
        self.target = PassData()
        self.domain_transforms = {"source": self.source,
                                  "target": self.target}
        self.deco = self.domain_transforms["source"]

    def get_decos(self, mode=None):
        return [("_source", self.source)]

    def get_deco_parameters(self):
        return self.source.parameters()


class SharedCombo(Combo):
    def __init__(self, deco_args, classifier, domain_classes=2, n_classes=10):
        super(SharedCombo, self).__init__(deco_args, classifier, domain_classes, n_classes)
        self._deconet = self.deco_architecture(deco_args)
        self.domain_transforms = {"source": self._deconet,
                                  "target": self._deconet}
        self.deco = self.domain_transforms["source"]

    def get_decos(self, mode=None):
        return [("", self.deco)]

    def get_deco_parameters(self):
        return self.deco.parameters()


class TargetOnlyCombo(Combo):
    def __init__(self, deco_args, classifier, domain_classes=2, n_classes=10):
        super(TargetOnlyCombo, self).__init__(deco_args, classifier, domain_classes, n_classes)
        self.source = PassData()
        self.target = self.deco_architecture(deco_args)
        self.domain_transforms = {"source": self.source,
                                  "target": self.target}
        self.deco = self.domain_transforms["source"]

    def get_decos(self, mode=None):
        return [("_target", self.target)]

    def get_deco_parameters(self):
        return self.target.parameters()


class BothCombo(Combo):
    def __init__(self, deco_args, classifier, domain_classes=2, n_classes=10):
        super(BothCombo, self).__init__(deco_args, classifier, domain_classes, n_classes)
        self.source = self.deco_architecture(deco_args)
        self.target = self.deco_architecture(deco_args)
        self.domain_transforms = {"source": self.source,
                                  "target": self.target}
        self.deco = self.domain_transforms["source"]

    def get_deco_parameters(self):
        return itertools.chain(self.source.parameters(), self.target.parameters())

    def get_decos(self, mode=None):
        if mode:
            return self.domain_transforms[mode]
        return ("_source", self.source), ("_target", self.target)


deco_modes = {"shared": SharedCombo,
              "separated": BothCombo,
              "source": SourceOnlyCombo,
              "target": TargetOnlyCombo}


class BasicDECO(nn.Module):
    def __init__(self, deco_args):
        super(BasicDECO, self).__init__()
        self.inplanes = deco_args.deco_kernels
        self.ratio = 1.0
        self.deco_args = deco_args
        if self.deco_args.no_residual:
            deco_args.train_deco_weight = False
            deco_args.train_image_weight = False
        if self.deco_args.train_deco_weight:
            self.deco_weight = Parameter(torch.FloatTensor(1), requires_grad=True)
        else:
            self.deco_weight = Variable(torch.FloatTensor(1)).cuda()
        if self.deco_args.train_image_weight:
            self.image_weight = Parameter(torch.FloatTensor(1), requires_grad=True)
        else:
            self.image_weight = Variable(torch.FloatTensor(1)).cuda()
        self.deco_weight.data.fill_(deco_args.deco_weight)
        self.image_weight.data.fill_(1.0)
        self.use_tanh = deco_args.use_tanh

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def weighted_sum(self, input_data, x):
        if self.deco_args.no_residual:
            self.ratio = input_data.norm() / x.norm()
            if self.use_tanh:
                x = torch.tanh(x)
            return x
        x = self.deco_weight * x
        input_data = self.image_weight * input_data
        self.ratio = input_data.norm() / x.norm()
        if self.use_tanh:
            return torch.tanh(x + input_data)
        else:
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
        if self.deco_args.no_pool:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=5, stride=4, padding=2,
                                   bias=False)
        else:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=5, stride=2, padding=2,
                                   bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(deco_args.block, self.inplanes, deco_args.n_layers)
        if self.deco_args.deconv:
            self.deconv = nn.ConvTranspose2d(self.inplanes * deco_args.block.expansion, deco_args.output_channels, 3, padding=1, stride=4)
        else:
            self.conv_out = nn.Conv2d(self.inplanes * deco_args.block.expansion, deco_args.output_channels, 1)
        self.init_weights()

    def forward(self, input_data):
        input_data = input_data.expand(input_data.data.shape[0], 3, input_data.data.shape[2], input_data.data.shape[3])
        x = self.conv1(input_data)
        x = self.bn1(x)
        x = self.relu(x)
        if self.deco_args.no_pool is False:
            x = self.maxpool(x)

        x = self.layer1(x)
        if self.deco_args.deconv:
            x = self.deconv(x, output_size=input_data.shape)
        else:
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

    def get_trainable_params(self):
        return self.parameters()


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


class AlexNetStyleDANN(BasicDANN):
    def get_trainable_params(self):
        return itertools.chain(self.domain_classifier.parameters(), self.class_classifier.parameters(),
                               self.bottleneck.parameters())


class AlexNet(AlexNetStyleDANN):
    def __init__(self, domain_classes, n_classes):
        super(AlexNet, self).__init__()
        pretrained = alexnet(pretrained=True)
        self._convs = pretrained.features
        self.bottleneck = nn.Linear(4096, 256)  # bottleneck
        self._classifier = nn.Sequential(
            Flatten(),
            nn.Dropout(),
            pretrained.classifier[1],  # nn.Linear(256 * 6 * 6, 4096),  #
            nn.ReLU(inplace=True),
            nn.Dropout(),
            pretrained.classifier[4],  # nn.Linear(4096, 4096),  #
            nn.ReLU(inplace=True),
            self.bottleneck,
            nn.ReLU(inplace=True)
        )
        self.features = nn.Sequential(self._convs, self._classifier)
        self.class_classifier = nn.Sequential(nn.Dropout(), nn.Linear(256, n_classes))
        self.domain_classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256, 1024),  # pretrained.classifier[1]
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 1024),  # pretrained.classifier[4]
            nn.ReLU(inplace=True),
            nn.Linear(1024, domain_classes),
        )


class CaffeNet(AlexNetStyleDANN):
    def __init__(self, domain_classes, n_classes):
        super(CaffeNet, self).__init__()
        pretrained = load_caffenet()
        self._convs = nn.Sequential(*list(pretrained)[:16])
        self.bottleneck = nn.Linear(4096, 256)  # bottleneck
        self._classifier = nn.Sequential(*list(pretrained)[16:22],
                                         self.bottleneck,
                                         nn.ReLU(inplace=True))
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


class AlexNetNoBottleneck(AlexNetStyleDANN):
    def __init__(self, domain_classes, n_classes):
        super(AlexNetNoBottleneck, self).__init__()
        pretrained = alexnet(pretrained=True)
        self._convs = pretrained.features
        self._classifier = nn.Sequential(
            Flatten(),
            nn.Dropout(),
            pretrained.classifier[1],  # nn.Linear(256 * 6 * 6, 4096),  #
            nn.ReLU(inplace=True),
            nn.Dropout(),
            pretrained.classifier[4],  # nn.Linear(4096, 4096),  #
            nn.ReLU(inplace=True),
        )
        self.features = nn.Sequential(self._convs, self._classifier)
        self.class_classifier = nn.Linear(4096, n_classes)
        self.domain_classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(4096, 1024),  # pretrained.classifier[1]
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 1024),  # pretrained.classifier[4]
            nn.ReLU(inplace=True),
            nn.Linear(1024, domain_classes),
        )

    def get_trainable_params(self):
        return itertools.chain(self.domain_classifier.parameters(), self.class_classifier.parameters())


classifier_list = {"roided_lenet": CNNModel,
                   "mnist": MnistModel,
                   "svhn": SVHNModel,
                   "alexnet": AlexNet,
                   "alexnet_no_bottleneck": AlexNetNoBottleneck,
                   "caffenet": CaffeNet}
