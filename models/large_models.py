import itertools

from torch import nn as nn
from torchvision.models import resnet50, alexnet
from torchvision.models.resnet import Bottleneck

from caffenet.caffenet_pytorch import load_caffenet
from models import torchvision_variants as tv
from models.model import BasicDECO, BasicDANN
from models.torch_future import Flatten


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
            self.deconv = nn.ConvTranspose2d(self.inplanes * deco_args.block.expansion, deco_args.output_channels, 5,
                                             padding=2, stride=4)
        else:
            self.conv_out = nn.Conv2d(self.inplanes * deco_args.block.expansion, deco_args.output_channels, 1)
        self.init_weights()

    def forward(self, input_data):
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


class BigDecoDANN(BasicDANN):
    def get_trainable_params(self):
        return itertools.chain(self.domain_classifier.parameters(), self.class_classifier.parameters(),
                               self.bottleneck.parameters())


class ResNet50(BigDecoDANN):
    def __init__(self, domain_classes, n_classes):
        super(ResNet50, self).__init__()
        resnet = resnet50(pretrained=True)
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool
        )
        self.class_classifier = nn.Linear(512 * Bottleneck.expansion, n_classes)
        self.domain_classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512 * Bottleneck.expansion, 1024),  # pretrained.classifier[1]
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 1024),  # pretrained.classifier[4]
            nn.ReLU(inplace=True),
            nn.Linear(1024, domain_classes),
        )

    def get_trainable_params(self):
        return itertools.chain(self.domain_classifier.parameters(), self.class_classifier.parameters())


class AlexNet(BigDecoDANN):
    def __init__(self, domain_classes, n_classes):
        super(AlexNet, self).__init__()
        pretrained = alexnet(pretrained=True)
        self.build_self(pretrained, domain_classes, n_classes)

    def build_self(self, pretrained, domain_classes, n_classes):
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


class SmallAlexNet(AlexNet):
    def __init__(self, domain_classes, n_classes):
        super(AlexNet, self).__init__()
        pretrained = tv.small_alexnet(pretrained=True)
        self.build_self(pretrained, domain_classes, n_classes)


class CaffeNet(BigDecoDANN):
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


class AlexNetNoBottleneck(BigDecoDANN):
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