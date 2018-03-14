import numpy as np
import os


def get_name(args, seed):
    name = "%s_lr:%g_BS:%d_epochs:%d_IS:%d_DannW:%g" % (args.optimizer, args.lr, args.batch_size, args.epochs,
                                                        args.image_size, args.DANN_weight)
    if args.entropy_loss_weight > 0.0:
        name += "_entropy:%g" % args.entropy_loss_weight
    if args.use_deco:
        name += "_deco%d_%d_%s_%dc" % (
            args.deco_blocks, args.deco_kernels, args.deco_block_type, args.deco_output_channels)
        if args.deco_bn:
            name += "_bn"
    else:
        name += "_vanilla"
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
