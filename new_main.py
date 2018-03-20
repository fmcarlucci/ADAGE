import argparse
import random

import time
import torch.backends.cudnn as cudnn
import torch.utils.data

from dataset import data_loader
from dataset.data_loader import get_dataloader
from logger import Logger
from models.model import classifier_list, get_net, deco_types
from test import test
from train.optim import get_optimizer_and_scheduler, optimizer_list, Optimizers
from train.utils import get_name, get_folder_name, ensure_dir, train_epoch


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
    parser.add_argument('--deco_blocks', default=4, type=int)
    parser.add_argument('--deco_kernels', default=64, type=int)
    parser.add_argument('--deco_block_type', default='basic', choices=deco_types.keys(),
                        help="Which kind of deco block to use")
    parser.add_argument('--deco_output_channels', type=int, default=3, help="3 or 1")
    # misc
    parser.add_argument('--suffix', help="Will be added to end of name", default="")
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
folder_name = get_folder_name(args.source, args.target)
logger = Logger("{}/{}/{}".format(log_folder, folder_name, run_name))

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

args.domain_classes = 1 + len(args.source)
dataloader_source = get_dataloader(args.source, batch_size, image_size, args.data_aug_mode)
dataloader_target = get_dataloader(args.target, batch_size, image_size, args.data_aug_mode)

# load model
my_net = get_net(args)

# setup optimizer
optimizer, scheduler = get_optimizer_and_scheduler(args.optimizer, my_net, args.epochs, args.lr, args.keep_pretrained_fixed)

if cuda:
    my_net = my_net.cuda()

start = time.time()
# training
for epoch in range(n_epoch):
    scheduler.step()
    logger.scalar_summary("aux/lr", scheduler.get_lr()[0], epoch)
    train_epoch(epoch, dataloader_source, dataloader_target, optimizer, my_net, logger, n_epoch, cuda, dann_weight,
                entropy_weight)
    for source in source_dataset_names:
        s_acc = test(source, epoch, my_net, image_size)
        if len(source_dataset_names) == 1:
            source_name = "acc/source"
        else:
            source_name = "acc/source_%s" % source
        logger.scalar_summary(source_name, s_acc, epoch)
    t_acc = test(target_dataset_name, epoch, my_net, image_size)
    logger.scalar_summary("acc/target", t_acc, epoch)

save_path = '{}/{}/{}_{}.pth'.format(model_root, folder_name, run_name, epoch)
print("Network saved to {}".format(save_path))
ensure_dir(save_path)
torch.save(my_net, save_path)
print('done, it took %g' % (time.time() - start))
