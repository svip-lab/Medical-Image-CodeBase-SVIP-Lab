# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       sinaDataLoader
   Project Name:    segOCT
   Author :         Administrator
   Date:            2018/11/3
-------------------------------------------------
   Change Activity:
                   2018/11/3:
-------------------------------------------------
"""
import sys
sys.path.append('./')
sys.path.append('../')

import os
import time
import pdb
from PIL import Image

import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms

import numpy as np

from utils.visualizer import Visualizer


class sinaLoader(nn.Container):
    def __init__(self, data_root, batch, num_val=165, num_worker=8):
        super(sinaLoader, self).__init__()
        # data_root: '/root/workspace/2018_OCT_transfer/dataset/sina/crop_mask/'
        # self.img_root=os.path.join(data_root, 'data_aug_images')
        # self.mask_root=os.path.join(data_root, 'data_aug_oct_mask')
        self.data_root = data_root
        self.batch = batch
        self.num_val=num_val
        self.workers = num_worker

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Resize((512, 512)),  # scale imported image
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                      std=[0.229, 0.224, 0.225]
            # )
        ]
        )  # transform it into a torch tensor

    def data_load(self):
        train_set = sinaTrainSet(data_root=self.data_root,
                                 num_val=self.num_val,
                                 transform=self.transform)
        val_set = sinaValSet(data_root=self.data_root,
                             num_val=self.num_val,
                             transform=self.transform)

        train_loader = data.DataLoader(
            dataset=train_set,
            batch_size=self.batch,
            shuffle=True,
            num_workers=self.workers,
            pin_memory=True
        )
        val_loader = data.DataLoader(
            dataset=val_set,
            batch_size=1,
            shuffle=True,
            num_workers=self.workers,
            pin_memory=True
        )

        return train_loader, val_loader


class sinaTrainSet(data.Dataset):
    def __init__(self, data_root, num_val, transform=None):
        super(sinaTrainSet, self).__init__()
        self.img_root = os.path.join(data_root, 'data_aug_images')
        self.mask_root = os.path.join(data_root, 'data_aug_oct_mask')
        assert len(os.listdir(self.img_root)) == len(os.listdir(self.mask_root)), 'wrong...'

        # all
        self.img_name_list = os.listdir(self.img_root)
        self.mask_name_list = os.listdir(self.mask_root)

        # train
        num_val = int(num_val)
        self.img_train_list = self.img_name_list[:-num_val]
        self.mask_train_list = self.mask_name_list[:-num_val]

        self.transform = transform

    def __len__(self):
        return len(self.img_train_list)

    def __getitem__(self, item):
        assert self.img_train_list[item] == self.mask_train_list[item], 'wrong...'
        img = Image.open(os.path.join(self.img_root, self.img_train_list[item]))
        mask = Image.open(os.path.join(self.mask_root, self.mask_train_list[item]))
        # mask = transforms.ToTensor()(mask/3.*255)
        mask = torch.from_numpy(np.array(mask, dtype=np.int)).long()
        if self.transform is not None:
            img = self.transform(img)
        return img, mask


class sinaValSet(sinaTrainSet):
    def __init__(self, data_root, num_val, transform=None):
        # img_root, mask_root, num_val, etc should be same in sinaTrainSet
        super(sinaValSet, self).__init__(data_root, num_val, transform)
        # val
        num_val=int(num_val)
        self.img_test_list = self.img_name_list[-num_val:]
        self.mask_test_list = self.mask_name_list[-num_val:]

    def __len__(self):
        return len(self.img_test_list)

    def __getitem__(self, item):
        assert self.img_test_list[item] == self.mask_test_list[item], 'wrong...'
        img = Image.open(os.path.join(self.img_root, self.img_test_list[item]))
        mask = Image.open(os.path.join(self.mask_root, self.mask_test_list[item]))

        # mask = transforms.ToTensor()(mask / 3. * 255)
        mask = torch.from_numpy(np.array(mask, dtype=np.int)).long()
        if self.transform is not None:
            img = self.transform(img)
        return img, mask


class sinaAnalyser(object):
    def __init__(self):
        self.vis = Visualizer(env='v99_sina', port=31432)
        data_root = '/root/workspace/2018_OCT_transfer/dataset/sina/crop_mask'
        # label_root = '/root/workspace/2018_OCT_transfer/dataset/sina/crop_mask/data_aug_label'
        # mask_root = '/root/workspace/2018_OCT_transfer/dataset/sina/crop_mask/data_aug_oct_mask'

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Resize((512, 512)),  # scale imported image
            transforms.ToTensor()])  # transform it into a torch tensor

        self.sina_set = sinaTrainSet(data_root=data_root, num_val=165)
        self.sina_val = sinaValSet(data_root=data_root, num_val=165)

    def run(self):
        for item in range(self.sina_set.__len__()):
            img, mask = self.sina_set.__getitem__(item)
            # img = self.transform(img).unsqueeze(0)
            print(img.shape)
            pdb.set_trace()
            if item % 10 == 0:
                print('hello sina TRAIN: {}'.format(item))
        for item in range(self.sina_val.__len__()):
            img, mask = self.sina_val.__getitem__(item)
            img = self.transform(img).unsqueeze(0)
            print(img.shape)
            if item % 10 == 0:
                print('hello sina VAL: {}'.format(item))
            # mask = np.array(mask)
            # mask = mask / 3. * 255
            # mask.astype(int)
            # self.vis.img_cpu(name='sina-img', img_=img)
            # self.vis.img_cpu(name='sina-border', img_=border)
            # self.vis.img_cpu(name='sina-mask', img_=mask)
            # self.vis.text(name)
            # time.sleep(1)


def main():
    import tb
    tb.colour()
    sinaAnalyser().run()

#     class PlaneDataset(data.Dataset):
#         def __init__(self, subset='train', downsample_rate=8, random_flip=False, root_dir=None):
#             assert subset in ['train', 'val']
#
#         self.MAX_PLANES = 20
#         self.subset = subset
#         self.random_flip = random_flip
#         self.root_dir = os.path.join(root_dir, subset)
#         self.downsample_rate = downsample_rate
#
#     self.img_transform = tf.Compose([
#         tf.Normalize(mean=[102.9801, 115.9465, 122.7717], std=[1., 1., 1.])
#     ])
#     data_paths = np.sort(os.listdir(os.path.join(root_dir, subset)))
#
#
# self.data_paths = list(map(lambda x: os.path.join(root_dir, subset, x), data_paths))
#
#
# def __getitem__(self, index):
#
#
#     """
# image: h x w x 3
# plane: plane parameters N x 3
# depth: h x w x 1
# normal: h x w x 3
# semantics: semantic segmentation h x w
# segmentation: plane instance segmentation h x w x 1
# boundary: h x w x 2
# num_planes: number of planes
# info: camera matrix 4 x 4 + 4
# """
# data = np.load(self.data_paths[index])
# # RGB image
# image = data['image']
# label = data['semantics'].astype(np.uint8)
# plane = data['segmentation'].squeeze().astype(np.uint8)
# image = image.astype(np.float32)
# if self.random_flip:
#     random_flip = np.random.choice([0, 1])
# if random_flip:
#     image = np.fliplr(image)
# label = np.fliplr(label)
# plane = np.fliplr(plane)
# image = image.transpose((2, 0, 1))
# if self.subset == 'train':
#     label = cv2.resize(label, (image.shape[2] // self.downsample_rate, image.shape[1] // self.downsample_rate),
#                        interpolation=cv2.INTER_NEAREST)
# plane = cv2.resize(plane, (image.shape[2] // self.downsample_rate, image.shape[1] // self.downsample_rate),
#                    interpolation=cv2.INTER_NEAREST)
# # ignore non-planar region
# label[plane == 0] = 0
# image = self.img_transform(torch.from_numpy(image.copy()))
# label = torch.from_numpy(label.astype(np.int)).long()
# sample = {'image': image, 'label': label}
# return sample
#
#
# def __len__(self):
#
#
#     return len(self.data_paths)


if __name__ == '__main__':
    main()
