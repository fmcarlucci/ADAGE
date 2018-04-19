import os

import numpy as np
import torch.utils
import torch.utils.data as data
from PIL import Image
from torchvision import datasets
from torchvision import transforms
from torchvision.datasets import ImageFolder

mnist = 'mnist'
mnist_m = 'mnist_m'
svhn = 'svhn'
synth = 'synth'
webcam = "webcam"
amazon = "amazon"
dslr = "dslr"
mnist_image_root = os.path.join('dataset', 'mnist')
mnist_m_image_root = os.path.join('dataset', 'mnist_m')
synth_image_root = os.path.join('dataset', 'SynthDigits')

office_list = [amazon, webcam, dslr]
dataset_list = [mnist, mnist_m, svhn, synth] + office_list


def get_images_for_conversion(folder_path, image_size=228):
    img_transform = get_transform(image_size, "test", None)
    return ImageFolderWithPath(folder_path, transform=img_transform)


def get_dataset(name, image_size, mode="train", limit=None):
    img_transform = get_transform(image_size, mode, name)
    dataset = load_dataset(img_transform, name, limit)
    return dataset


def get_transform(image_size, mode, name):
    if mode == "train":
        img_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
    elif mode == "office":
        img_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif mode == "simple":
        img_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.9, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
    elif mode == "test":
        if name in office_list:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        else:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
        img_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    return img_transform


index_cache = {}
def load_dataset(img_transform, dataset_name, limit=None):
    if dataset_name == mnist:
        dataset = datasets.MNIST(
            root=mnist_image_root,
            train=True,
            transform=img_transform, download=True
        )
    elif dataset_name == svhn:
        dataset = datasets.SVHN(
            root=os.path.join('dataset', 'svhn'),
            transform=img_transform, download=True
        )
    elif dataset_name == mnist_m:
        train_list = os.path.join(mnist_m_image_root, 'mnist_m_train_labels.txt')
        dataset = GetLoader(
            data_root=os.path.join(mnist_m_image_root, 'mnist_m_train'),
            data_list=train_list,
            transform=img_transform
        )
    elif dataset_name == synth:
        train_mat = os.path.join(synth_image_root, 'synth_train_32x32.mat')
        dataset = GetSynthDigits(
            data_root=synth_image_root,
            data_mat=train_mat,
            transform=img_transform
        )
    elif dataset_name == amazon:
        dataset = datasets.ImageFolder('dataset/amazon', transform=img_transform)
    elif dataset_name == dslr:
        dataset = datasets.ImageFolder('dataset/dslr', transform=img_transform)
    elif dataset_name == webcam:
        dataset = datasets.ImageFolder('dataset/webcam', transform=img_transform)
    elif type(dataset_name) is list:
        return ConcatDataset([load_dataset(img_transform, dset, limit) for dset in dataset_name])
    if limit:
        indices = index_cache.get((dataset_name, limit), None)
        if indices is None:
            indices = torch.randperm(len(dataset))[:limit]
        index_cache[(dataset_name, limit)] = indices
        dataset = Subset(dataset, indices)
    return RgbWrapper(dataset)


class GetLoader(data.Dataset):
    def __init__(self, data_root, data_list, transform=None):
        self.root = data_root
        self.transform = transform

        f = open(data_list, 'r')
        data_list = f.readlines()
        f.close()

        self.n_data = len(data_list)

        self.img_paths = []
        self.img_labels = []

        for data in data_list:
            self.img_paths.append(data[:-3])
            self.img_labels.append(data[-2])

    def __getitem__(self, item):
        img_paths, labels = self.img_paths[item], self.img_labels[item]
        imgs = Image.open(os.path.join(self.root, img_paths)).convert('RGB')

        if self.transform is not None:
            imgs = self.transform(imgs)
            labels = int(labels)

        return imgs, labels

    def __len__(self):
        return self.n_data


class GetSynthDigits(data.Dataset):
    def __init__(self, data_root, data_mat, transform=None):
        self.root = data_root
        self.transform = transform

        import scipy.io as sio

        # reading(loading) mat file as array
        loaded_mat = sio.loadmat(data_mat)
        self.data = loaded_mat['X']
        self.data = np.transpose(self.data, (3, 2, 0, 1))
        self.labels = loaded_mat['y'].astype(np.int64).squeeze()

    def __getitem__(self, item):
        img, labels = self.data[item], self.labels[item]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)
            labels = int(labels)

        return img, labels

    def __len__(self):
        return len(self.data)


def get_dataloader(dataset_name, batch_size, image_size, mode, limit):
    return torch.utils.data.DataLoader(
        dataset=get_dataset(dataset_name, image_size, mode, limit),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True)


class RgbWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, i):
        data, label = self.dataset.__getitem__(i)
        return data.expand(3, data.shape[1], data.shape[2]), label


class ImageFolderWithPath(ImageFolder):
    def __getitem__(self, index):
        path, _ = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, path


class Subset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i % len(d)] for d in self.datasets)

    def __len__(self):
        return max(len(d) for d in self.datasets)
