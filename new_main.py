import random

import time
import torch.backends.cudnn as cudnn
import torch.utils.data

from dataset.data_loader import get_dataloader, get_subdataloader
from logger import Logger
from models.model import get_net
from test import test
from train.optim import get_optimizer_and_scheduler
from train.utils import get_name, get_folder_name, ensure_dir, train_epoch, get_args, do_pretraining

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
test_batch_size = 1024
if image_size > 100:
    test_batch_size = 256
n_epoch = args.epochs
dann_weight = args.DANN_weight
entropy_weight = args.entropy_loss_weight

source_dataset_names = args.source
target_dataset_name = args.target

random.seed(manual_seed)
torch.manual_seed(manual_seed)

args.domain_classes = 1 + len(args.source)
dataloader_source = get_dataloader(args.source, batch_size, image_size, args.data_aug_mode, args.source_limit)
dataloader_target = get_dataloader(args.target, batch_size, image_size, args.data_aug_mode, args.target_limit)

# load model
my_net = get_net(args)

# setup optimizer
optimizer, scheduler = get_optimizer_and_scheduler(args.optimizer, my_net, args.epochs, args.lr,
                                                   args.keep_pretrained_fixed)

if cuda:
    my_net = my_net.cuda()

if args.deco_pretrain > 0:
    do_pretraining(args.deco_pretrain, dataloader_source, dataloader_target, my_net, logger)
start = time.time()
# training
for epoch in range(n_epoch):
    scheduler.step()
    logger.scalar_summary("aux/lr", scheduler.get_lr()[0], epoch)
    train_epoch(epoch, dataloader_source, dataloader_target, optimizer, my_net, logger, n_epoch, cuda, dann_weight,
                entropy_weight, scheduler)
    my_net.set_deco_mode("source")
    for source in source_dataset_names:
        s_acc = test(source, epoch, my_net, image_size, test_batch_size)
        if len(source_dataset_names) == 1:
            source_name = "acc/source"
        else:
            source_name = "acc/source_%s" % source
        logger.scalar_summary(source_name, s_acc, epoch)
    my_net.set_deco_mode("target")
    t_acc = test(target_dataset_name, epoch, my_net, image_size, test_batch_size)
    logger.scalar_summary("acc/target", t_acc, epoch)

save_path = '{}/{}/{}_{}.pth'.format(model_root, folder_name, run_name, epoch)
print("Network saved to {}".format(save_path))
ensure_dir(save_path)
torch.save(my_net, save_path)
print('done, it took %g' % (time.time() - start))
