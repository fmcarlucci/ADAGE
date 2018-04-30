import math

import itertools
import torch
import torch.nn as nn
from torch import nn as nn
from torch.autograd import Function, Variable
from torch.nn import Parameter
from torchvision.models.resnet import BasicBlock, Bottleneck
import torch.nn.functional as func

from models.torch_future import Flatten

image_weight = 1.0
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


def get_classifier(name, domain_classes, n_classes, generalization):
    if name:
        return classifier_list[name](domain_classes, n_classes, generalization)
    return CNNModel(domain_classes, n_classes)


def get_net(args):
    domain_classes = args.domain_classes
    if args.use_deco:
        deco_args = DecoArgs(args)
        my_net = deco_modes[deco_args.mode](deco_args, classifier=args.classifier, domain_classes=domain_classes,
                                            n_classes=args.n_classes, generalization=args.generalization)
    else:
        my_net = get_classifier(args.classifier, domain_classes=domain_classes, n_classes=args.n_classes, generalization=args.generalization)

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


class GradientKillerLayer(Function):
    @staticmethod
    def forward(ctx, x, **kwargs):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return None, None


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
    def __init__(self, deco_args, classifier, domain_classes=2, n_classes=10, generalization=False):
        super(Combo, self).__init__()
        self.net = get_classifier(classifier, domain_classes, n_classes, generalization)
        from models.large_models import SmallAlexNet, BigDecoDANN, DECO
        if isinstance(self.net, SmallAlexNet):
            self.deco_architecture = DECO_mini
        elif isinstance(self.net, BigDecoDANN):
            self.deco_architecture = DECO
        else:
            self.deco_architecture = DECO_mini

    def set_deco_mode(self, mode):
        self.deco = self.domain_transforms[mode]

    def forward(self, input_data, lambda_val, domain):
        input_data = self.deco(input_data)
        return self.net(input_data, lambda_val, domain)

    def get_trainable_params(self):
        return itertools.chain(self.get_deco_parameters(), self.net.get_trainable_params())


class NoDecoCombo(Combo):
    def __init__(self, deco_args, classifier, domain_classes=2, n_classes=10):
        super(NoDecoCombo, self).__init__(deco_args, classifier, domain_classes, n_classes)
        self.deco = PassData

    def set_deco_mode(self, mode):
        pass

    def get_trainable_params(self):
        return self.net.get_trainable_params()


class SourceOnlyCombo(Combo):
    def __init__(self, deco_args, classifier, domain_classes=2, n_classes=10):
        super(SourceOnlyCombo, self).__init__(deco_args, classifier, domain_classes, n_classes)
        self.source = self.deco_architecture(deco_args)
        self.target = PassData()
        self.domain_transforms = {"source": self.source,
                                  "target": self.target}
        self.deco = self.domain_transforms["source"]

    def get_decos(self, mode=None):
        return [("source", self.source)]

    def get_deco_parameters(self):
        return self.source.parameters()


class SharedCombo(Combo):
    def __init__(self, deco_args, classifier, domain_classes=2, n_classes=10, generalization=False):
        super(SharedCombo, self).__init__(deco_args, classifier, domain_classes, n_classes, generalization)
        self._deconet = self.deco_architecture(deco_args)
        self.domain_transforms = {"source": self._deconet,
                                  "target": self._deconet}
        self.deco = self.domain_transforms["source"]

    def get_decos(self, mode=None):
        return [("shared", self.deco)]

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
        return [("target", self.target)]

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
        return ("source", self.source), ("target", self.target)


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
        self.image_weight.data.fill_(image_weight)
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
        self.conv_out = nn.Conv2d(self.inplanes, deco_args.output_channels, 1)
        self.init_weights()

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.conv_out(x)

        return self.weighted_sum(input_data, x)


class Tiny_DECO(BasicDECO):
    def __init__(self, deco_args):
        super(Tiny_DECO, self).__init__(deco_args)
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=5, stride=4, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(deco_args.block, self.inplanes, deco_args.n_layers)
        self.conv_out = nn.Conv2d(self.inplanes * deco_args.block.expansion, deco_args.output_channels, 1)
        self.init_weights()

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.conv_out(x)

        return self.weighted_sum(input_data, x)


class BasicDANN(nn.Module):
    def __init__(self):
        super(BasicDANN, self).__init__()
        self.features = None
        self.domain_classifier = None
        self.class_classifier = None
        self.observer = PassData()

    def forward(self, input_data, lambda_val, domain=None):
        feature = self.features(input_data)
        feature = feature.view(input_data.shape[0], -1)
        reverse_feature = ReverseLayerF.apply(feature, lambda_val)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)
        observation = self.observer(GradientKillerLayer.apply(feature))
        return class_output, domain_output, observation

    def get_trainable_params(self):
        return self.parameters()

    # TODO: after refactoring, remove this
    def set_deco_mode(self, mode):
        pass


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


class MultisourceModelWeighted(BasicDANN):
    def __init__(self, domain_classes, n_classes, generalization):
        super(MultisourceModelWeighted, self).__init__()
        self.domains = domain_classes
        if generalization:
            self.domains -= 1
        self.generalization = generalization
        self.n_classes = n_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        self.class_classifier = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(True),
            Flatten(),
            nn.Linear(256 * 4 * 4, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 1024),
            nn.ReLU(True)
        )
        self.per_domain_classifier = nn.ModuleList([nn.Linear(1024, n_classes) for k in range(domain_classes - 1)])
        self.domain_classifier = nn.Sequential(
            Flatten(),
            nn.Linear(256 * 4 * 4, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 2048),
            nn.ReLU(True),
            nn.Linear(2048, self.domains)
        )
        self.observer = nn.Sequential(
            Flatten(),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, self.domains)
        )

    def forward(self, input_data, lambda_val, domain):
        feature = self.features(input_data)
        reverse_feature = ReverseLayerF.apply(feature, lambda_val)
        class_features = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)
        observation = self.observer(GradientKillerLayer.apply(feature))
        if domain < len(self.per_domain_classifier):  # one of the source domains
            class_output = self.per_domain_classifier[domain](class_features)
        else:  # if target domain
            class_output = Variable(torch.zeros(input_data.shape[0], self.n_classes).cuda())

            if self.generalization:
                softmax_obs = nn.functional.softmax(observation, 1)
            else:
                softmax_obs = nn.functional.softmax(observation[:, :-1], 1)
            for k, predictor in enumerate(self.per_domain_classifier):
                class_output = class_output + nn.functional.softmax(predictor(class_features), 1) * Variable(softmax_obs[:, k].mean().data, requires_grad=False)
        return class_output, domain_output, observation


class MultisourceModel(BasicDANN):
    def __init__(self, domain_classes, n_classes, generalization):
        super(MultisourceModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        self.class_classifier = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(True),
            Flatten(),
            nn.Linear(256 * 4 * 4, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 1024),
            nn.ReLU(True),
            nn.Linear(1024, n_classes)
        )
        self.domain_classifier = nn.Sequential(
            Flatten(),
            nn.Linear(256 * 4 * 4, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 2048),
            nn.ReLU(True),
            nn.Linear(2048, domain_classes)
        )
        self.observer = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            Flatten(),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, domain_classes)
        )

    def forward(self, input_data, lambda_val, domain=None):
        feature = self.features(input_data)
        reverse_feature = ReverseLayerF.apply(feature, lambda_val)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)
        observation = self.observer(GradientKillerLayer.apply(input_data))
        return class_output, domain_output, observation


class CNNModel(BasicDANN):
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
        feature = self.feature(input_data)
        feature = feature.view(-1, 50 * 4 * 4)
        reverse_feature = ReverseLayerF.apply(feature, lambda_val)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output


from models.large_models import DECO, BigDecoDANN, ResNet50, AlexNet, SmallAlexNet, CaffeNet, AlexNetNoBottleneck

classifier_list = {"roided_lenet": CNNModel,
                   "mnist": MnistModel,
                   "svhn": SVHNModel,
                   "multi": MultisourceModel,
                   "multi_weighted": MultisourceModelWeighted,
                   "alexnet": AlexNet,
                   "alexnet_no_bottleneck": AlexNetNoBottleneck,
                   "caffenet": CaffeNet,
                   "small_alexnet": SmallAlexNet,
                   "resnet50": ResNet50}
