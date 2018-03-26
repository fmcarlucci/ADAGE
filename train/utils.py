import argparse

import numpy as np
import os

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn

from dataset import data_loader
from models.model import entropy_loss, Combo, deco_types, classifier_list, deco_modes
from train.optim import optimizer_list, Optimizers, get_optimizer_and_scheduler
import itertools


def get_name(args, seed):
    name = "%s_lr:%g_BS:%d_epochs:%d_IS:%d_DannW:%g_DA%s" % (args.optimizer, args.lr, args.batch_size, args.epochs,
                                                             args.image_size, args.DANN_weight, args.data_aug_mode)
    if args.keep_pretrained_fixed:
        name += "_pretrainedFixed"
    if args.entropy_loss_weight > 0.0:
        name += "_entropy:%g" % args.entropy_loss_weight
    if args.use_deco:
        name += "_deco%d_%d_%s_%dc" % (
            args.deco_blocks, args.deco_kernels, args.deco_block_type, args.deco_output_channels)
        if args.deco_mode != "shared":
            name += "_" + args.deco_mode
        if args.deco_no_residual:
            name += "_no_residual"
        if args.deco_tanh:
            name += "_tanh"
        elif args.train_deco_weight or args.train_image_weight:
            name += "_train%s%sWeight" % (
                "Deco" if args.train_deco_weight else "", "Image" if args.train_image_weight else "")
    else:
        name += "_vanilla"
    if args.classifier:
        name += "_" + args.classifier
    if args.suffix:
        name += "_" + args.suffix
    return name + "_%d" % (seed)


def to_np(x):
    return x.data.cpu().numpy()


def to_grid(x):
    channels = x.shape[1]
    s = x.shape[2]
    y = x.swapaxes(1, 3).reshape(3, s * 3, s, channels).swapaxes(1, 2).reshape(s * 3, s * 3, channels).squeeze()[
        np.newaxis, ...]
    return y


def get_folder_name(source, target):
    return '-'.join(source) + "_" + target


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def do_pretraining(num_epochs, dataloader_source, dataloader_target, model, logger, mode="shared"):
    optimizer, scheduler = get_optimizer_and_scheduler(Optimizers.adam.value, model, num_epochs, 0.01, True)
    loss_f = nn.BCELoss(size_average=False).cuda()
    for epoch in range(num_epochs):
        model.train()
        if len(dataloader_source) > len(dataloader_target):
            source_loader = dataloader_source
            target_loader = itertools.cycle(dataloader_target)
        else:
            source_loader = itertools.cycle(dataloader_source)
            target_loader = dataloader_target

        for i, (source_batches, target_data) in enumerate(zip(source_loader, target_loader)):
            scheduler.step()
            optimizer.zero_grad()
            source_loss = 0.0
            import ipdb; ipdb.set_trace()
            if mode != "target":
                model.set_deco_mode("source")
                for v, source_data in enumerate(source_batches):
                    s_img, _ = source_data
                    img_in = Variable(s_img).cuda()
                    out = model.deco(img_in)
                    loss = loss_f((out/2.0)+0.5, (img_in/2.0)+0.5)
                    loss.backward()
                    source_loss += loss.cpu().numpy()

            # pretrain target deco only if needed
            target_loss = 0.0
            if mode != "source":
                model.set_deco_mode("target")
                target_image, _ = target_data
                img_in = Variable(target_image).cuda()
                out = model.deco(img_in)
                loss = loss_f((out/2.0)+0.5, (img_in/2.0)+0.5)
                loss.backward()
                target_loss = loss.cpu().numpy()
            optimizer.step()
            if i == 0:
                source_images = Variable(s_img[:9], volatile=True).cuda()
                target_images = Variable(target_image[:9], volatile=True).cuda()
                model.set_deco_mode("source")
                source_images = model.deco(source_images)
                model.set_deco_mode("target")
                target_images = model.deco(target_images)
                logger.image_summary("reconstruction/source", to_grid(to_np(source_images)), epoch)
                logger.image_summary("reconstruction/target", to_grid(to_np(target_images)), epoch)

        print("Reconstruction loss source: %g, target %g" % (source_loss, target_loss))
        logger.scalar_summary("reconstruction/source", source_loss, epoch)
        logger.scalar_summary("reconstruction/target", target_loss, epoch)


def train_epoch(epoch, dataloader_source, dataloader_target, optimizer, model, logger, n_epoch, cuda,
                dann_weight, entropy_weight):
    model.train()
    len_dataloader = min(len(dataloader_source), len(dataloader_target))
    data_sources_iter = iter(dataloader_source)
    data_target_iter = iter(dataloader_target)

    batch_idx = 0
    while batch_idx < len_dataloader:
        absolute_iter_count = batch_idx + epoch * len_dataloader
        p = float(absolute_iter_count) / n_epoch / len_dataloader
        lambda_val = 2. / (1. + np.exp(-10 * p)) - 1
        data_sources_batch = data_sources_iter.next()
        # process source datasets (can be multiple)
        err_s_label = 0.0
        err_s_domain = 0.0
        num_source_domains = len(data_sources_batch)
        model.set_deco_mode("source")
        for v, source_data in enumerate(data_sources_batch):
            s_img, s_label = source_data
            class_loss, domain_loss = compute_batch_loss(cuda, lambda_val, model, s_img, s_label, v + 1)
            loss = class_loss + dann_weight * domain_loss
            loss.backward()
            # used for logging only
            err_s_label += class_loss.data.cpu().numpy()
            err_s_domain += domain_loss.data.cpu().numpy()
        err_s_label = err_s_label / num_source_domains
        err_s_domain = err_s_domain / num_source_domains

        # training model using target data
        model.set_deco_mode("target")
        t_img, _ = data_target_iter.next()
        entropy_target, err_t_domain = compute_batch_loss(cuda, lambda_val, model, t_img, None, 0)
        loss = entropy_weight * entropy_target * lambda_val + dann_weight * err_t_domain
        loss.backward()

        # err = dann_weight * err_t_domain + dann_weight * err_s_domain + err_s_label + entropy_weight * entropy_target * lambda_val
        optimizer.step()
        optimizer.zero_grad()

        # logging stuff
        if batch_idx is 0:
            source_images = Variable(s_img[:9], volatile=True).cuda()
            target_images = Variable(t_img[:9], volatile=True).cuda()
            if isinstance(model, Combo):
                model.set_deco_mode("source")
                source_images = model.deco(source_images)
                model.set_deco_mode("target")
                target_images = model.deco(target_images)
                for prefix, deco in model.get_decos():
                    logger.scalar_summary("aux/deco_to_image_ratio" + prefix, deco.ratio.data.cpu()[0], epoch)
                    logger.scalar_summary("aux/deco_weight" + prefix, deco.deco_weight.data.cpu()[0], epoch)
                    logger.scalar_summary("aux/image_weight" + prefix, deco.image_weight.data.cpu()[0], epoch)
            logger.image_summary("images/source", to_grid(to_np(source_images)), epoch)
            logger.image_summary("images/target", to_grid(to_np(target_images)), epoch)
            logger.scalar_summary("aux/p", p, epoch)
            logger.scalar_summary("aux/lambda", lambda_val, epoch)

        if (batch_idx % (len_dataloader / 2 + 1)) == 0:
            logger.scalar_summary("loss/source", err_s_label, absolute_iter_count)
            logger.scalar_summary("loss/domain", (err_s_domain + err_t_domain) / 2, absolute_iter_count)
            logger.scalar_summary("loss/entropy_target", entropy_target, absolute_iter_count)
            print('epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_domain: %f' \
                  % (epoch, batch_idx, len_dataloader, err_s_label,
                     err_s_domain, err_t_domain.cpu().data.numpy()))
        batch_idx += 1


def compute_reconstruction_loss(input_image, model):
    loss = F.binary_cross_entropy(size_average=False)


def compute_batch_loss(cuda, lambda_val, model, img, label, domain_label):
    domain_label = torch.ones(img.shape[0]).long() * domain_label
    if cuda:
        img = img.cuda()
        if label is not None: label = label.cuda()
        domain_label = domain_label.cuda()
    class_output, domain_output = model(input_data=Variable(img), lambda_val=lambda_val)
    # compute losses
    if label is not None:
        class_loss = F.cross_entropy(class_output, Variable(label))
    else:
        class_loss = entropy_loss(class_output)
    domain_loss = F.cross_entropy(domain_output, Variable(domain_label))
    return class_loss, domain_loss


def get_args():
    parser = argparse.ArgumentParser()
    # optimizer
    parser.add_argument('--optimizer', choices=optimizer_list, default=Optimizers.adam.value)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--keep_pretrained_fixed', action="store_true")
    # data
    parser.add_argument('--image_size', type=int, default=28)
    parser.add_argument('--data_aug_mode', default="train", choices=["train", "simple", "office"])
    parser.add_argument('--source', default=[data_loader.mnist], choices=data_loader.dataset_list, nargs='+')
    parser.add_argument('--target', default=data_loader.mnist_m, choices=data_loader.dataset_list)
    parser.add_argument('--n_classes', default=10, type=int)
    # losses
    parser.add_argument('--DANN_weight', default=1.0, type=float)
    parser.add_argument('--entropy_loss_weight', default=0.0, type=float, help="Entropy loss on target, default is 0")
    # deco
    parser.add_argument('--use_deco', action="store_true", help="If true use deco architecture")
    parser.add_argument('--train_deco_weight', default=True, type=bool, help="Train the deco weight (True by default)")
    parser.add_argument('--train_image_weight', default=False, type=bool,
                        help="Train the image weight (False by default)")
    parser.add_argument('--deco_no_residual', action="store_true", help="If set, no residual will be applied to DECO")
    parser.add_argument('--deco_blocks', default=4, type=int)
    parser.add_argument('--deco_kernels', default=64, type=int)
    parser.add_argument('--deco_block_type', default='basic', choices=deco_types.keys(),
                        help="Which kind of deco block to use")
    parser.add_argument('--deco_output_channels', type=int, default=3, help="3 or 1")
    parser.add_argument('--deco_mode', default="shared", choices=deco_modes.keys())
    parser.add_argument('--deco_tanh', action="store_true", help="If set, tanh will be applied to DECO output")
    # misc
    parser.add_argument('--suffix', help="Will be added to end of name", default="")
    parser.add_argument('--classifier', default=None, choices=classifier_list.keys())
    parser.add_argument('--tmp_log', action="store_true", help="If set, logger will save to /tmp instead")
    return parser.parse_args()
