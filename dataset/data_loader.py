import torch.utils.data as data
from PIL import Image
import os
from torchvision import datasets
from torchvision import transforms

source_dataset_name = 'mnist'
target_dataset_name = 'mnist_m'
mnist_image_root = os.path.join('dataset', 'mnist')
mnist_m_image_root = os.path.join('dataset', 'mnist_m')


def get_dataset(name, image_size, mode="train"):
    if mode is "train":
        img_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.3, 1.0)),
            # transforms.RandomCrop(image_size),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
    else:
        img_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    if name is "mnist":
        dataset = datasets.MNIST(
            root=mnist_image_root,
            train=True,
            transform=img_transform, download=True
        )
    if name is "svhn":
        dataset = datasets.SVHN(
            root=os.path.join('dataset', 'svhn'),
            train=True,
            transform=img_transform, download=True
        )
    elif name is "mnist_m":
        train_list = os.path.join(mnist_m_image_root, 'mnist_m_train_labels.txt')

        dataset = GetLoader(
            data_root=os.path.join(mnist_m_image_root, 'mnist_m_train'),
            data_list=train_list,
            transform=img_transform
        )
    return dataset


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
