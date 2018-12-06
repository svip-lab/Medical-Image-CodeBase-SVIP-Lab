# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       ultrasoundDataLoader
   Project Name:    clsUS
   Author :         Kang ZHOU
   Date:            2018/11/28
-------------------------------------------------
   Change Activity:
                   2018/11/28:
-------------------------------------------------
"""
import sys
sys.path.append('../')

import os
from PIL import Image

import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as T
from torchvision.datasets.folder import *

import numpy as np
import h5py


class ultraDataLoader(object):
    # image classification dataloader
    def __init__(self, data_root, batch, num_test=400, workers=8):
        super(ultraDataLoader, self).__init__()
        self.data_root = data_root
        self.batch = batch
        self.workers = workers

        self.num_test = num_test

        # images transform
        # data augumentation
        self.transform = None
        self.target_transform = None

    def data_load(self):
        train_set = ultraDataSet(data_root=self.data_root,
                                 mode='train',
                                 transform=self.transform,
                                 target_transform=self.target_transform)
        test_set = ultraDataSet(data_root=self.data_root,
                                 mode='test',
                                 transform=self.transform,
                                 target_transform=self.target_transform)
        # data_set = ultraDataSet(data_root=self.data_root,
        #                          mode='train',
        #                          transform=self.transform,
        #                          target_transform=self.target_transform)
        # __indices = np.arange(data_set.__len__()).tolist()
        # train_set = data.dataset.Subset(data_set, __indices[:-self.num_test])
        # test_set = data.dataset.Subset(data_set, __indices[-self.num_test:])

        train_loader = data.DataLoader(
            dataset=train_set,
            batch_size=self.batch,
            # shuffle=True,
            num_workers=self.workers,
            pin_memory=True
        )
        test_loader = data.DataLoader(
            dataset=test_set,
            batch_size=1,
            num_workers=self.workers,
            pin_memory=True
        )

        return train_loader, test_loader


class ultraDataSet(data.Dataset):
    # image classification dataset from folder
    # mat file
    def __init__(self, data_root, mode='train', transform=None, target_transform=None):
        super(ultraDataSet, self).__init__()
        # dataroot: '/root/workspace/2018_US_project/ultrsound_zhy'

        extensions = ['.mat']
        train_root = os.path.join(data_root, mode)
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

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((448, 448))
            # T.Resize((224, 224))
        ])
        self.norm = T.Normalize(
                # mean=[0.485, 0.456, 0.406],
                # std=[0.229, 0.224, 0.225]
                mean=[0.485], std=[0.229]
            )
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
        # ['bad', 'good', 'none'] -> [0, 1, 2]
        # {'bad': 0, 'good': 1, 'none': 2}
        class_to_idx = {classes[i]: i for i in range(len(classes))}

        # multi classes
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        name = path.split('/')[-1]
        with h5py.File(path, 'r') as f:
            sample = f['img']
            sample = np.expand_dims(sample, 2)

            if self.transform is not None:
                sample = self.transform(sample)
                sample = T.ToTensor()(sample)
                sample = sample.transpose(1, 2)
                sample = self.norm(sample)
                # sample = torch.Tensor(np.array(sample).transpose())
            # target = torch.Tensor(target)
            # target = torch.Tensor(np.array(self.targets[index])).long()
            target = torch.Tensor(np.array(target)).long()
            return sample, target, name

    def __len__(self):
        return len(self.samples)


def data_analysis():
    import time
    from utils.visualizer import Visualizer
    vis = Visualizer(env='{}'.format('v99_99_debug'), port=31434)
    dataroot = '/root/workspace/2018_US_project/ultrsound_zhy'
    train_loader, test_loader = ultraDataLoader(dataroot, 1).data_load()

    for i, (sample, target, name) in enumerate(train_loader):
        # sample.shape = (B, 1, 480, 480)
        vis.images(sample)
        vis.plot_single_win(dict(max=sample.max(), mean=sample.mean(), min=sample.min()),
                            win='sample')
        vis.plot_multi_win(dict(target=target.item()))
        time.sleep(2)

def data_debug():
    path = '/root/workspace/2018_US_project/ultrsound_zhy/test/good/us32_280.mat'
    with h5py.File(path, 'r') as f:
        sample = f['img']
        # sample = np.array(sample)
        sample = np.expand_dims(sample, 2)
        print(sample.dtype)
        print('max={}, mean={:.3f}, shape={}'.format(np.array(sample).max(), np.array(sample).mean(), sample.shape))
        sample = T.ToPILImage()(sample)
        print('max={}, mean={:.3f}, shape={}'.format(np.array(sample).max(), np.array(sample).mean(), sample.size))
        sample = T.Resize((480, 480))(sample)
        print('max={}, mean={:.3f}, shape={}'.format(np.array(sample).max(), np.array(sample).mean(), sample.size))
        sample = sample.convert('L')
        print('max={}, mean={:.3f}, shape={}'.format(np.array(sample).max(), np.array(sample).mean(), sample.size))
        # sample = T.ToTensor()(sample)
        sample = torch.Tensor(np.array(sample))
        print('max={}, mean={:.3f}, shape={}'.format(np.array(sample).max(), np.array(sample).mean(), sample.shape))
        # target = torch.Tensor(target)



def main():
    # user code

    data_analysis()
    # data_debug()




if __name__ == '__main__':
    # pdb: python debuger
    import pdb
    # tb(Traceback.colour) function should be removed
    import tb

    tb.colour()
    main()
