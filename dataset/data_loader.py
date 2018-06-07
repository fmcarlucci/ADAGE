import os
import gzip
import pickle

import numpy as np
import torch.utils
import torch.utils.data as data
from PIL import Image
from torchvision import datasets
from torchvision import transforms
from torchvision.datasets import ImageFolder

from train.utils import simple_tuned

mnist = 'mnist'
mnist_m = 'mnist_m'
svhn = 'svhn'
synth = 'synth'
usps = 'usps'

synth_signs = "synth_signs"
gtsrb = "gtsrb"
webcam = "webcam"
amazon = "amazon"
dslr = "dslr"
eth80_p00 = "eth80-p-000"
eth80_p22 = "eth80-p-022"
eth80_p45 = "eth80-p-045"
eth80_p68 = "eth80-p-068"
eth80_p90 = "eth80-p-090"
eth_list = [eth80_p00, eth80_p22, eth80_p45, eth80_p68, eth80_p90]

mnist_image_root = os.path.join('dataset', 'mnist')
mnist_m_image_root = os.path.join('dataset', 'mnist_m')
synth_image_root = os.path.join('dataset', 'SynthDigits')
usps_image_root = os.path.join('dataset', 'usps')
gtsrb_image_root = os.path.join('dataset', gtsrb, "signs_")
synth_signs_image_root = os.path.join('dataset', synth_signs, "synth_signs_")

office_list = [amazon, webcam, dslr]
dataset_list = [mnist, mnist_m, svhn, synth, usps, synth_signs, gtsrb] + office_list + eth_list

dataset_std = {mnist: (0.30280363, 0.30280363, 0.30280363),
               mnist_m: (0.2384788, 0.22375608, 0.24496263),
               svhn: (0.1951134, 0.19804622, 0.19481073),
               synth: (0.29410212, 0.2939651, 0.29404707),
               usps: (0.25887518, 0.25887518, 0.25887518),
               synth_signs: (0.28315591, 0.2789663, 0.30152685),
               gtsrb: (0.26750497, 0.2494335,  0.25966593)}

dataset_mean = {mnist: (0.13909429, 0.13909429, 0.13909429),
                mnist_m: (0.45920207, 0.46326601, 0.41085603),
                svhn: (0.43744073, 0.4437959, 0.4733686),
                synth: (0.46332872, 0.46316052, 0.46327512),
                usps: (0.17025368, 0.17025368, 0.17025368),
                synth_signs: (0.43471373, 0.40261434, 0.43641199),
                gtsrb: (0.36089954, 0.31854348, 0.33791215)}


def get_images_for_conversion(folder_path, image_size=228):
    img_transform = get_transform(image_size, "test", None)
    return ImageFolderWithPath(folder_path, transform=img_transform)


def get_dataset(dataset_name, image_size, mode="train", limit=None):
    img_transform = get_transform(image_size, mode, dataset_name)
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
    elif dataset_name == usps:
        data_file = "usps_28x28.pkl"
        dataset = GetUSPS(
            data_root=usps_image_root,
            data_file=data_file,
            transform=img_transform
        )
    elif dataset_name == gtsrb:
        dataset = GetNumpyDataset(gtsrb_image_root, mode, img_transform)
    elif dataset_name == synth_signs:
        dataset = GetNumpyDataset(synth_signs_image_root, mode, img_transform)
    elif dataset_name == amazon:
        dataset = datasets.ImageFolder('dataset/amazon', transform=img_transform)
    elif dataset_name == dslr:
        dataset = datasets.ImageFolder('dataset/dslr', transform=img_transform)
    elif dataset_name == webcam:
        dataset = datasets.ImageFolder('dataset/webcam', transform=img_transform)
    elif dataset_name in eth_list:
        dataset = datasets.ImageFolder('dataset/' + dataset_name, transform=img_transform)
    elif type(dataset_name) is list:
        return ConcatDataset([get_dataset(dset, image_size, mode, limit) for dset in dataset_name])
    if limit:
        indices = index_cache.get((dataset_name, limit), None)
        if indices is None:
            indices = torch.randperm(len(dataset))[:limit]
        index_cache[(dataset_name, limit)] = indices
        dataset = Subset(dataset, indices)
    return RgbWrapper(dataset)


def get_transform(image_size, mode, name):
    if isinstance(name, list):
        return None
    # TODO use dataset specific mean and std
    if mode == "train":
        img_transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.5, hue=0.5),
            transforms.RandomAffine(15, shear=15),
            transforms.RandomResizedCrop(image_size, scale=(0.9, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=dataset_mean[name], std=dataset_std[name])
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
    elif mode == simple_tuned:
        img_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.9, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=dataset_mean[name], std=dataset_std[name])
        ])
    elif mode == "simple-no-norm":
        img_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.9, 1.0)),
            transforms.ToTensor()
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
            transforms.Normalize(mean=dataset_mean[name], std=dataset_std[name])
        ])
    elif mode == "test-tuned":
        img_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=dataset_mean[name], std=dataset_std[name])
        ])
    return img_transform


index_cache = {}

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


class GetNumpyDataset(data.Dataset):
    def __init__(self, data_root, mode, transform=None):
        self.root = data_root
        self.transform = transform

        test = np.load(self.root + "test_data.npy")
        l_test = np.load(self.root + "test_labels.npy")
        if mode in ["test", "test-tuned"]:
            self.data = test
            self.labels = l_test
        else:
            train = np.load(self.root + "train_data.npy")
            l_train = np.load(self.root + "train_labels.npy")
            self.data = np.vstack((train, test))
            self.labels = np.hstack((l_train, l_test))

        self.labels = self.labels.astype(np.int64).squeeze()

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


class GetUSPS(data.Dataset):
    def __init__(self, data_root, data_file, transform=None):
        self.root = data_root
        self.filename = data_file
        # Num of Train = 7438, Num ot Test 1860
        self.transform = transform
        self.dataset_size = None
        self.data, self.labels = self.load_samples()

        total_num_samples = self.labels.shape[0]
        indices = np.arange(total_num_samples)
        np.random.shuffle(indices)
        self.data = self.data[indices[0:self.dataset_size], ::]
        self.labels = self.labels[indices[0:self.dataset_size]].astype(np.int64).squeeze()
        self.data *= 255.0
        self.data = np.repeat(self.data.astype("uint8"), 3, axis=1)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        img, labels = self.data[index, ::], self.labels[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
            labels = int(labels)
        # label = torch.FloatTensor([label.item()])
        return img, labels

    def load_samples(self):
        """Load sample images from dataset."""
        filename = os.path.join(self.root, self.filename)
        f = gzip.open(filename, "rb")
        data_set = pickle.load(f, encoding="bytes")
        f.close()
        images_train = data_set[0][0]
        images_test = data_set[1][0]
        images = np.concatenate((images_train, images_test), axis=0)
        labels_train = data_set[0][1]
        labels_test = data_set[1][1]
        labels = np.concatenate((labels_train, labels_test), axis=0)
        self.dataset_size = labels.shape[0]
        return images, labels

    def __len__(self):
        """Return size of dataset."""
        return self.dataset_size


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
