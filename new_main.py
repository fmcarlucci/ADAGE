import argparse
import random

import numpy as np
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.autograd import Variable

from dataset import data_loader
from dataset.data_loader import get_dataloader
from logger import Logger
from models.model import classifier_list, entropy_loss, get_net, deco_types
from test import test
from train.optim import get_optimizer_and_scheduler, optimizer_list, Optimizers
from train.utils import get_name, to_np, to_grid, get_folder_name


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
    parser.add_argument('--source', default=[data_loader.mnist], choices=data_loader.dataset_list)
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
logger = Logger("{}/{}/{}".format(log_folder, get_folder_name(args.source, args.target), run_name))

model_root = 'models'

cuda = True
cudnn.benchmark = True
lr = args.lr
batch_size = args.batch_size
image_size = args.image_size
n_epoch = args.epochs
dann_weight = args.DANN_weight
entropy_weight = args.entropy_loss_weight

source_dataset_names = args.source
target_dataset_name = args.target

random.seed(manual_seed)
torch.manual_seed(manual_seed)

dataloader_source = get_dataloader(args.source, batch_size, image_size)
dataloader_target = get_dataloader(args.target, batch_size, image_size)

# load model
my_net = get_net(args)

# setup optimizer
optimizer, scheduler = get_optimizer_and_scheduler(args.optimizer, my_net, args.epochs, args.lr)

loss_class = torch.nn.CrossEntropyLoss()
loss_domain = torch.nn.CrossEntropyLoss()

if cuda:
    my_net = my_net.cuda()
    loss_class = loss_class.cuda()
    loss_domain = loss_domain.cuda()

# training
for epoch in range(n_epoch):
    if scheduler:
        scheduler.step()
    my_net.train(True)
    len_dataloader = min(len(dataloader_source), len(dataloader_target))
    data_source_iter = iter(dataloader_source)
    data_target_iter = iter(dataloader_target)
    batch_idx = 0
    while batch_idx < len_dataloader:
        absolute_iter_count = batch_idx + epoch * len_dataloader
        p = float(absolute_iter_count) / n_epoch / len_dataloader
        lambda_val = 2. / (1. + np.exp(-10 * p)) - 1
        optimizer.zero_grad()
        data_source_batch = data_source_iter.next()
        # process source datasets
        source_class_loss = Variable(torch.zeros(1), requires_grad=True)
        source_domain_loss = Variable(torch.zeros(1), requires_grad=True)
        if cuda:
            source_class_loss = source_class_loss.cuda()
            source_domain_loss = source_domain_loss.cuda()
        for v, source_data in enumerate(data_source_batch):
            source_domain_label = torch.ones(batch_size).long() * (v+1)
            s_img, s_label = source_data
            if cuda:
                s_img = s_img.cuda()
                s_label = s_label.cuda()
                source_domain_label = source_domain_label.cuda()
            class_output, domain_output = my_net(input_data=Variable(s_img), lambda_val=lambda_val)
            source_class_loss += loss_class(class_output, Variable(s_label))
            source_domain_loss += loss_domain(domain_output, Variable(source_domain_label))
        err_s_label = source_class_loss / len(data_source_batch)
        err_s_domain = source_domain_loss / len(data_source_batch)

        # training model using target data
        target_domain_label = torch.zeros(batch_size).long()
        t_img, _ = data_target_iter.next()
        if cuda:
            t_img = t_img.cuda()
            target_domain_label = target_domain_label.cuda()
        target_class_output, domain_output = my_net(input_data=Variable(t_img), lambda_val=lambda_val)
        err_t_domain = loss_domain(domain_output, Variable(target_domain_label))
        entropy_target = entropy_loss(target_class_output)

        # global error
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

            logger.scalar_summary("loss/entropy_target", entropy_target, absolute_iter_count)
            print('epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_domain: %f' \
                  % (epoch, batch_idx, len_dataloader, err_s_label.cpu().data.numpy(),
                     err_s_domain.cpu().data.numpy(), err_t_domain.cpu().data.numpy()))
        batch_idx += 1

    my_net.train(False)
    for source in source_dataset_names:
        s_acc = test(source, epoch, my_net, image_size)
        if len(source_dataset_names) == 0:
            source_name = ""
        else:
            source_name = source
        logger.scalar_summary("acc/source%s" % source_name, s_acc, absolute_iter_count)
    t_acc = test(target_dataset_name, epoch, my_net, image_size)
    logger.scalar_summary("acc/target", t_acc, absolute_iter_count)
    logger.scalar_summary("aux/p", p, absolute_iter_count)
    logger.scalar_summary("aux/lambda", lambda_val, absolute_iter_count)

save_path = '{}/{}_{}/{}_{}.pth'.format(model_root, args.source, args.target, run_name, epoch)
print("Network saved to {}".format(save_path))
torch.save(my_net, save_path)
print('done')
