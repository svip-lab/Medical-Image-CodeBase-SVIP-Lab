# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       ultrasoundDataLoader
   Project Name:    segOCT
   Author :         Kang ZHOU
   Date:            2018/11/16
-------------------------------------------------
   Change Activity:
                   2018/11/16:
-------------------------------------------------
"""
import sys

import os
from PIL import Image
import pdb

import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as T
from torchvision.datasets.folder import *


class ultraLoader(nn.Container):
    def __init__(self, root, batch, version, num_worker=8):
        super(ultraLoader, self).__init__()
        self.root = root
        assert version in ['train', 'validation', 'test_ours', 'bigan', 'cyclegan'], 'wrong'
        self.version = version
        if version == 'train':
            self.batch = batch
        else:
            self.batch = 1
        self.workers = num_worker

        # images transform
        # data augumentation
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor()])

    def data_load(self):
        train_set = ultraTrainSet(self.root, IMG_EXTENSIONS, self.version, transform=self.transform)
        train_loader = data.DataLoader(
            dataset=train_set,
            batch_size=self.batch,
            shuffle=True,
            num_workers=self.workers,
            pin_memory=True
        )
        return train_loader

class ultraTestLoader(nn.Container):
    def __init__(self, root, version, num_worker=8):
        super(ultraTestLoader, self).__init__()
        self.root = root
        self.version = version
        self.workers = num_worker

        # images transform
        # data augumentation
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor()])

    def data_load(self):
        test_set = ultraTestSet(self.root, self.version, self.transform)
        test_loader = data.DataLoader(
            dataset=test_set,
            batch_size=1,
            shuffle=True,
            num_workers=self.workers,
            pin_memory=True
        )
        return test_loader


class ultraTrainSet(data.Dataset):
    def __init__(self, dataroot, extensions, version, transform=None, target_transform=None):
        super(ultraTrainSet, self).__init__()
        # dataroot: '/root/workspace/2018_OCT_transfer/dataset/ultrasound'

        train_root = os.path.join(dataroot, version)
        classes, class_to_idx = self._find_classes(train_root)

        samples = make_dataset(train_root, class_to_idx, extensions)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " +
                                train_root + "\n" "Supported extensions are: " + ",".join(
                extensions)))

        self.train_root = train_root
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

        self.transform = transform
        self.target_transform = target_transform


    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        # according to alphabetical order: ['bad', 'good']
        classes.sort()

        # folder name -> label
        # ['bad', 'good'] -> [0, 1]
        # {'bad': 0, 'good': 1}
        class_to_idx = {classes[i]: i for i in range(len(classes))}

        # multi classes
        # class_to_idx = {classes[i]: (0 if classes[i] == 'bad' else 1) for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        # sample = self.loader(path)
        sample = Image.open(path)
        sample = sample.convert('L')
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

    # def __repr__(self):
    #     fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
    #     fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
    #     fmt_str += '    Root Location: {}\n'.format(self.train_root)
    #     tmp = '    Transforms (if any): '
    #     fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
    #     tmp = '    Target Transforms (if any): '
    #     fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
    #     return fmt_str



class ultraTestSet(data.Dataset):
    def __init__(self, dataroot, version, transform):
        super(ultraTestSet, self).__init__()
        # dataroot: '/root/workspace/2018_OCT_transfer/dataset/ultrasound'
        self.img_dir = os.path.join(dataroot, 'test', version)
        self.transform = transform
        # self.img_name_list = os.listdir(self.img_dir)
        self.img_name_list = [f.name for f in os.scandir(self.img_dir) if f.is_file()]
        self.target = torch.ones(self.__len__())

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, item):
        img_name = self.img_name_list[item]
        img = Image.open(os.path.join(self.img_dir, self.img_name_list[item]))
        target = self.target[item]

        img = img.convert('L')
        if self.transform is not None:
            img = self.transform(img)

        return img, target, img_name


def main():
    # tb(Traceback.colour) function should be removed
    import tb
    tb.colour()

    # user code
    ultraTrainSet()


if __name__ == '__main__':
    main()
