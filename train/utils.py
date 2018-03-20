import numpy as np
import os

import torch
from torch.autograd import Variable
import torch.nn.functional as F

from models.model import entropy_loss, Combo


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
        if args.deco_no_residual:
            name += "no_residual"
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


def train_epoch(epoch, dataloader_source, dataloader_target, optimizer, model, logger, n_epoch, cuda,
                dann_weight, entropy_weight):
    model.train()
    len_dataloader = min(len(dataloader_source), len(dataloader_target))
    data_sources_iter = iter(dataloader_source)
    data_target_iter = iter(dataloader_target)

    batch_idx = 0
    # import ipdb; ipdb.set_trace()
    while batch_idx < len_dataloader:
        absolute_iter_count = batch_idx + epoch * len_dataloader
        p = float(absolute_iter_count) / n_epoch / len_dataloader
        lambda_val = 2. / (1. + np.exp(-10 * p)) - 1

        data_sources_batch = data_sources_iter.next()
        # process source datasets (can be multiple)
        err_s_label = 0.0
        err_s_domain = 0.0
        num_source_domains = len(data_sources_batch)
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
                source_images = model.deco(source_images)
                target_images = model.deco(target_images)
                logger.scalar_summary("aux/deco_to_image_ratio", model.deco.ratio.data.cpu()[0], epoch)
                logger.scalar_summary("aux/deco_weight", model.deco.deco_weight.data.cpu()[0], epoch)
                logger.scalar_summary("aux/image_weight", model.deco.image_weight.data.cpu()[0], epoch)
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
