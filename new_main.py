import argparse
import itertools
import random

import numpy as np
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.autograd import Variable
from torchvision.models.resnet import BasicBlock, Bottleneck

from dataset import data_loader
from dataset.data_loader import get_dataloader
from logger import Logger
from models.model import Combo, classifier_list, get_classifier, entropy_loss
from test import test
from train.optim import get_optimizer_and_scheduler, optimizer_list, Optimizers
from train.utils import get_name, to_np, to_grid

deco_types = {'basic': BasicBlock, 'bottleneck': Bottleneck}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimizer', choices=optimizer_list, default=Optimizers.adam.value)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--image_size', type=int, default=28)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--DANN_weight', default=1.0, type=float)
    parser.add_argument('--entropy_loss_weight', default=0.0, type=float, help="Entropy loss on target, default is 0")
    parser.add_argument('--use_deco', action="store_true", help="If true use deco architecture")
    parser.add_argument('--train_deco_weight', action="store_true", help="Train the deco weight")
    parser.add_argument('--deco_blocks', default=4, type=int)
    parser.add_argument('--deco_kernels', default=64, type=int)
    parser.add_argument('--deco_block_type', default='basic', choices=deco_types.keys(),
                        help="Which kind of deco block to use")
    parser.add_argument('--deco_bn', action="store_true", help="If set, deco output will be normalized")
    parser.add_argument('--deco_output_channels', type=int, default=3, help="3 or 1")
    parser.add_argument('--suffix', help="Will be added to end of name", default="")
    parser.add_argument('--sources', default=[data_loader.mnist], choices=data_loader.dataset_list)
    parser.add_argument('--target', default=data_loader.mnist_m, choices=data_loader.dataset_list)
    parser.add_argument('--classifier', default=None, choices=classifier_list.keys())
    parser.add_argument('--tmp_log', action="store_true", help="If set, logger will save to /tmp instead")
    return parser.parse_args()


args = get_args()
print(args)
manual_seed = random.randint(1, 1000)
run_name = get_name(args, manual_seed)
print("Working on " + run_name)
log_folder = "logs/"
if args.tmp_log:
    log_folder = "/tmp/"
logger = Logger("{}/{}_{}/{}".format(log_folder, args.source, args.target, run_name))

model_root = 'models'

cuda = True
cudnn.benchmark = True
lr = args.lr
batch_size = args.batch_size
image_size = args.image_size
n_epoch = args.epochs
dann_weight = args.DANN_weight
entropy_weight = args.entropy_loss_weight

source_dataset_name = args.source
target_dataset_name = args.target

random.seed(manual_seed)
torch.manual_seed(manual_seed)

dataloader_source = get_dataloader(args.source, batch_size, image_size)
dataloader_target = get_dataloader(args.target, batch_size, image_size)

# load model

if args.use_deco:
    my_net = Combo(n_deco=args.deco_blocks, classifier=args.classifier, train_deco_weight=args.train_deco_weight,
                   deco_bn=args.deco_bn, deco_kernels=args.deco_kernels, deco_block=deco_types[args.deco_block_type],
                   out_channels=args.deco_output_channels)
else:
    my_net = get_classifier(args.classifier)

# setup optimizer
optimizer, scheduler = get_optimizer_and_scheduler(args.optimizer, my_net, args.epochs, args.lr)

loss_class = torch.nn.CrossEntropyLoss()
loss_domain = torch.nn.CrossEntropyLoss()

if cuda:
    my_net = my_net.cuda()
    loss_class = loss_class.cuda()
    loss_domain = loss_domain.cuda()

for p in my_net.parameters():
    p.requires_grad = True

exit()
# training
for epoch in range(n_epoch):
    if scheduler:
        scheduler.step()
    # this must be done each epoch, or zip will exhaust
    if len(dataloader_source) > len(dataloader_target):
        len_dataloader = len(dataloader_source)
        combined_loader = zip(dataloader_source, itertools.cycle(dataloader_target))
    else:
        len_dataloader = len(dataloader_target)
        combined_loader = zip(itertools.cycle(dataloader_source), dataloader_target)
    my_net.train(True)
    for batch_idx, (source_batch, target_batch) in enumerate(combined_loader):
        absolute_iter_count = batch_idx + epoch * len_dataloader
        source_domain_label = torch.zeros(batch_size).long()
        target_domain_label = torch.ones(batch_size).long()
        s_img, s_label = source_batch
        t_img, _ = target_batch
        if cuda:
            s_img = s_img.cuda()
            s_label = s_label.cuda()
            t_img = t_img.cuda()
            source_domain_label = source_domain_label.cuda()
            target_domain_label = target_domain_label.cuda()

        p = float(absolute_iter_count) / n_epoch / len_dataloader
        lambda_val = 2. / (1. + np.exp(-10 * p)) - 1  # TODO: consider changing 10 to 2 to have lambda=1 for longer

        optimizer.zero_grad()

        class_output, domain_output = my_net(input_data=Variable(s_img), lambda_val=lambda_val)
        err_s_label = loss_class(class_output, Variable(s_label))
        err_s_domain = loss_domain(domain_output, Variable(source_domain_label))
        entropy_source = entropy_loss(class_output)
        # training model using target data

        target_class_output, domain_output = my_net(input_data=Variable(t_img), lambda_val=lambda_val)
        err_t_domain = loss_domain(domain_output, Variable(target_domain_label))
        entropy_target = entropy_loss(target_class_output)
        err = dann_weight * err_t_domain + dann_weight * err_s_domain + err_s_label + entropy_weight * entropy_target * lambda_val
        err.backward()
        optimizer.step()

        if batch_idx is 0:
            if args.use_deco:
                source_images = my_net.deco(Variable(s_img[:9]))
                target_images = my_net.deco(Variable(t_img[:9]))
                logger.scalar_summary("aux/deco_to_image_ratio", my_net.deco.ratio.data.cpu()[0], absolute_iter_count)
                logger.scalar_summary("aux/deco_weight", my_net.deco.deco_weight.data.cpu()[0], absolute_iter_count)
            else:
                source_images = Variable(s_img[:9])
                target_images = Variable(t_img[:9])
            logger.image_summary("images/source", to_grid(to_np(source_images)), absolute_iter_count)
            logger.image_summary("images/target", to_grid(to_np(target_images)), absolute_iter_count)
            if scheduler:
                logger.scalar_summary("aux/lr", scheduler.get_lr()[0], absolute_iter_count)

        if (batch_idx % (len_dataloader / 2 + 1)) == 0:
            logger.scalar_summary("loss/source", err_s_label, absolute_iter_count)
            logger.scalar_summary("loss/domain", (err_s_domain + err_t_domain) / 2, absolute_iter_count)
            # logger.scalar_summary("loss/entropy_source", entropy_source, absolute_iter_count)
            logger.scalar_summary("loss/entropy_target", entropy_target, absolute_iter_count)
            print('epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_domain: %f' \
                  % (epoch, batch_idx, len_dataloader, err_s_label.cpu().data.numpy(),
                     err_s_domain.cpu().data.numpy(), err_t_domain.cpu().data.numpy()))

    my_net.train(False)
    s_acc = test(source_dataset_name, epoch, my_net, image_size)
    t_acc = test(target_dataset_name, epoch, my_net, image_size)
    logger.scalar_summary("acc/source", s_acc, absolute_iter_count)
    logger.scalar_summary("acc/target", t_acc, absolute_iter_count)
    logger.scalar_summary("aux/p", p, absolute_iter_count)
    logger.scalar_summary("aux/lambda", lambda_val, absolute_iter_count)

save_path = '{}/{}_{}/{}_{}.pth'.format(model_root, args.source, args.target, run_name, epoch)
print("Network saved to {}".format(save_path))
torch.save(my_net, save_path)
print('done')
