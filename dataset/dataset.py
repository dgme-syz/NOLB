import os
from PIL import Image

import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset

# 处理数据集

## 1.  获取数据集并处理为可用于torch张量的形式

def get_dataset(args):
    args.dataset = args.dataset.lower()

    data_name_list = ['cifar10', 'cifar100']
    assert args.dataset in data_name_list, print('The name of dataset is not correct! Please input one from: ',
                                                 data_name_list)
    print("Warning: You are using {} dataset!".format(args.dataset))

    if args.dataset.startswith('cifar'):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomApply(transforms_list, p=0.5)
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        if args.dataset == 'cifar10':
            train_dataset = IMBALANCECIFAR10(root=args.data_path,
                                             imb_type=args.imb_type,
                                             imb_factor=args.imb_factor,
                                             rand_number=args.rand_number,
                                             train=True,
                                             download=True,
                                             transform=transform_train)
            val_dataset = datasets.CIFAR10(root=args.data_path,
                                           train=False,
                                           download=True,
                                           transform=transform_val)
        elif args.dataset == 'cifar100':
            train_dataset = IMBALANCECIFAR100(root=args.data_path,
                                              imb_type=args.imb_type,
                                              imb_factor=args.imb_factor,
                                              rand_number=args.rand_number,
                                              train=True,
                                              download=True,
                                              transform=transform_train)
            val_dataset = datasets.CIFAR100(root=args.data_path,
                                            train=False,
                                            download=True,
                                            transform=transform_val)
        else:
            raise ValueError('Please input the correct name of CIFAR10 or CIFAR100!')

        cls_num_list = train_dataset.get_cls_num_list()
        print('cls num list (first 10):', cls_num_list[:10])

    else:
        raise ValueError('Please input the correct name of CIFAR10 or CIFAR100!')
    return train_dataset, val_dataset, cls_num_list


## 2. 处理CIFAR-10,获取相关信息，并将其处理为长尾数据
class IMBALANCECIFAR10(torchvision.datasets.CIFAR10):
    cls_num = 10

    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(IMBALANCECIFAR10, self).__init__(root, train, transform, target_transform, download)
        np.random.seed(rand_number)

        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.gen_imbalanced_data(img_num_list)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list


## 2. 处理CIFAR-100,获取相关信息
class IMBALANCECIFAR100(IMBALANCECIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    cls_num = 100


if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = IMBALANCECIFAR10(root='./data', train=True, download=True, transform=transform)

    trainloader = iter(trainset)
    data, label = next(trainloader)
