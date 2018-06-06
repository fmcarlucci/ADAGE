import os
from argparse import ArgumentParser
from os import path

import torch
from torchvision.utils import save_image

from dataset.data_loader import get_images_for_conversion


def get_args():
    args = ArgumentParser()
    args.add_argument("model_path")
    args.add_argument("input_path")
    args.add_argument("output_path")
    return args.parse_args()


def convert_dataset(model, input_loader, output_folder, input_prefix):
    for i, (img, im_path) in enumerate(input_loader):
        out = torch.tanh(model.deco(img.unsqueeze(0).cuda())).squeeze().data
        outpath = path.join(output_folder, im_path[input_prefix:])
        folder = path.dirname(outpath)
        if not path.exists(folder):
            os.makedirs(folder)
        save_image(out, outpath)
        if i % 100 == 0:
            print("%d/%d" % (i, len(input_loader)))


if __name__ == "__main__":
    args = get_args()
    input_folder = args.input_path
    l = len(input_folder)
    output_folder = args.output_path
    model_path = args.model_path
    model = torch.load(model_path)
    input_loader = get_images_for_conversion(input_folder, image_size=256)
    with torch.no_grad():
        convert_dataset(model, input_loader, output_folder, l)
