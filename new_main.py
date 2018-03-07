import argparse
import random
import itertools
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

from dataset import data_loader
from dataset.data_loader import get_dataset

from logger import Logger
from models.model import CNNModel, Combo, classifier_list, get_classifier
import numpy as np
from test import test
import time


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--image_size', type=int, default=28)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--DANN_weight', default=1.0, type=float)
    parser.add_argument('--use_deco', action="store_true", help="If true use deco architecture")
    parser.add_argument('--train_deco_weight', action="store_true")
    parser.add_argument('--suffix', help="Will be added to end of name", default="")
    parser.add_argument('--source', default="mnist", choices=data_loader.dataset_list)
    parser.add_argument('--target', default="mnist_m", choices=data_loader.dataset_list)
    parser.add_argument('--classifier', default=None, choices=classifier_list.keys())
    parser.add_argument('--tmp_log', action="store_true", help="If set, logger will save to /tmp instead")
    return parser.parse_args()


def get_name(args, seed):
    name = "lr:%g_BS:%d_epochs:%d_DannW:%g_IS:%d" % (args.lr, args.batch_size, args.epochs,
                                                     args.DANN_weight, args.image_size)
    if args.use_deco:
        name += "_deco"
    if args.train_deco_weight:
        name += "_trainWeight"
    if args.classifier:
        name += "_" + args.classifier
    if args.suffix:
        name += "_" + args.suffix
    return name + "_%d" % (seed)


def to_np(x):
    return x.data.cpu().numpy()


def to_grid(x):
    channels = x.shape[1]
    y = x.swapaxes(1, 3).reshape(3, 28 * 3, 28, channels).swapaxes(1, 2).reshape(28 * 3, 28 * 3, channels).squeeze()[
        np.newaxis, ...]
    print(y.shape)
    return y


args = get_args()
manual_seed = random.randint(1, 1000)
run_name = get_name(args, manual_seed)
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
source_dataset_name = args.source
target_dataset_name = args.target

random.seed(manual_seed)
torch.manual_seed(manual_seed)

dataloader_source = torch.utils.data.DataLoader(
    dataset=get_dataset(args.source, image_size),
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=4)

dataloader_target = torch.utils.data.DataLoader(
    dataset=get_dataset(args.target, image_size),
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=4)

# load model

if args.use_deco:
    my_net = Combo(classifier=args.classifier, train_deco_weight=args.train_deco_weight)
else:
    my_net = get_classifier(args.classifier)

# setup optimizer
optimizer = optim.Adam(my_net.parameters(), lr=lr)

loss_class = torch.nn.NLLLoss()
loss_domain = torch.nn.NLLLoss()

if cuda:
    my_net = my_net.cuda()
    loss_class = loss_class.cuda()
    loss_domain = loss_domain.cuda()

for p in my_net.parameters():
    p.requires_grad = True

# training
for epoch in range(n_epoch):
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

        # training model using target data

        _, domain_output = my_net(input_data=Variable(t_img), lambda_val=lambda_val)
        err_t_domain = loss_domain(domain_output, Variable(target_domain_label))
        err = dann_weight * err_t_domain + dann_weight * err_s_domain + err_s_label
        err.backward()
        optimizer.step()

        if batch_idx is 0:
            if args.use_deco:
                source_images = my_net.deco(Variable(s_img[:9]))
                target_images = my_net.deco(Variable(t_img[:9]))
            else:
                source_images = Variable(s_img[:9])
                target_images = Variable(t_img[:9])
            logger.image_summary("images/source", to_grid(to_np(source_images)), absolute_iter_count)
            logger.image_summary("images/target", to_grid(to_np(target_images)), absolute_iter_count)

        if (batch_idx % 200) == 0:
            logger.scalar_summary("loss/source", err_s_label, absolute_iter_count)
            logger.scalar_summary("loss/domain", (err_s_domain + err_t_domain) / 2, absolute_iter_count)
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
    if args.use_deco:
        logger.scalar_summary("aux/deco_weight", my_net.deco.deco_weight.data.cpu()[0], absolute_iter_count)

save_path = '{}/{}_{}/{}_{}.pth'.format(model_root, args.source, args.target, run_name, epoch)
print("Network saved to {}".format(save_path))
torch.save(my_net, save_path)
print('done')
