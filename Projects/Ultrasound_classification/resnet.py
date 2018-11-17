# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       main_v10_Resnet
   Project Name:    segOCT
   Author :         Kang ZHOU
   Date:            2018/11/16
-------------------------------------------------
   Change Activity:
                   2018/11/16:
-------------------------------------------------
"""
import os
import datetime

import numpy as np
import sklearn.metrics as metrics

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim


from data.ultrasoundDataLoader import *
from networks.resnet import resnet50
from utils.visualizer import Visualizer
from utils.trick import *
from tool.parser_v10 import ParserArgs

class ResnetRunner(object):
    def __init__(self):
        args = ParserArgs().args
        cuda_visible(args.gpu)

        model = resnet50(in_channels=1, num_classes=2)
        model = nn.DataParallel(model).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                     weight_decay=args.weight_decay)

        # Optionally resume from a checkpoint
        if args.resume:
            ckpt_root = os.path.join('/root/workspace', args.project, 'checkpoints')
            ckpt_path = os.path.join(ckpt_root, args.resume)
            if os.path.isfile(ckpt_path):
                print("=> loading checkpoint '{}'".format(args.resume))
            #     checkpoint = torch.load(ckpt_path)
            #     args.start_epoch = checkpoint['epoch']
            #     self.val_best_iou = checkpoint['best_iou']
            #     model.load_state_dict(checkpoint['state_dict'])
            #     optimizer.load_state_dict(checkpoint['optimizer'])
            #     print("=> loaded checkpoint '{}' (epoch {})"
            #           .format(args.resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))

        cudnn.benchmark = True

        self.vis = Visualizer(env='{}'.format(args.version), port=args.port)

        self.train_loader = ultraLoader(root=args.dataroot, batch=args.batch, version='train').data_load()
        self.val_loader = ultraLoader(root=args.dataroot, batch=args.batch, version='validation').data_load()
        self.test_loader = ultraLoader(root=args.dataroot, batch=args.batch, version='test_ours').data_load()
        self.test_loader_bigan = ultraLoader(root=args.dataroot, batch=args.batch, version='bigan').data_load()
        self.test_loader_cyclegan = ultraLoader(root=args.dataroot, batch=args.batch, version='cyclegan').data_load()

        print_args(args)
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.criterion = nn.CrossEntropyLoss().cuda()


    def train_test(self):
        self.best_acc = 0
        for epoch in range(self.args.n_epochs):
            adjust_lr(self.args.lr, self.optimizer, epoch, 30)
            self.epoch = epoch

            self.train()
            self.test(self.val_loader, 'validation')
            self.test(self.test_loader, 'test_ours')
            self.test(self.test_loader_bigan, 'bigan')
            self.test(self.test_loader_cyclegan, 'cyclegan')
            print('\n', '*' * 10, 'Program Information', '*' * 10)
            print('Node: {}'.format(self.args.node))
            print('Version: {}\n'.format(self.args.version))


    def train(self):
        self.model.train()
        for i, (img, label) in enumerate(self.train_loader):
            img = img.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)

            output = self.model(img)
            _, pred = torch.max(output, 1)

            loss = self.criterion(output, label)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if i % 2 == 0:
                self.vis.images(img[0].squeeze(), name='train', img_name='{}_{}'
                                .format(label[0].item(), pred[0].item()))

            if i+1 == self.train_loader.__len__():
                self.vis.plot_many(dict(loss=loss.item()))
            if i % self.args.print_freq == 0:
                print('[{}] Epoch: [{}][{}/{}]\t, Loss: {:.4f}'.format
                      (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                       self.epoch, i, self.train_loader.__len__(), loss))


    def test(self, test_loader, version):
        prob_list = []
        pred_list = []
        true_list = []

        self.model.eval()
        with torch.no_grad():
            for i, (img, label) in enumerate(test_loader):
                img = img.cuda(non_blocking=True)
                label = label.cuda(non_blocking=True)

                output = self.model(img)
                output = F.softmax(output, dim=1)
                _, pred = torch.max(output, 1)

                prob_list.append(output[0][1].item())
                pred_list.append(pred.item())
                true_list.append(label.item())

                if i % 3 == 0:
                    self.vis.images(img.squeeze(), name=version, img_name='{}_{}'
                                    .format(label.item(), label.item()))

            # fpr, tpr, thresholds = metrics.roc_curve(
            #     y_true=true_list, y_score=prob_list, pos_label=1, drop_intermediate=False)
            #
            # pdb.set_trace()
            # auc = metrics.auc(fpr, tpr)
            auc = metrics.roc_auc_score(y_true=true_list, y_score=prob_list)
            acc = metrics.accuracy_score(y_true=true_list, y_pred=pred_list)

            if version == 'validation':
                is_best = acc > self.best_acc
                self.best_acc = max(acc, self.best_acc)
                save_ckpt(version=self.args.version,
                          state={
                              'epoch': self.epoch + 1,
                              'state_dict': self.model.state_dict(),
                              'best_acc': self.best_acc,
                              'optimizer': self.optimizer.state_dict(),
                          },
                          is_best=is_best,
                          epoch=self.epoch + 1,
                          project='2018_OCT_transfer')
                print('Save ckpt successfully!')

            print('*' * 10, 'Auc = {:.3f}, Acc = {:.3f}'.format(auc, acc), '*' * 10)
            self.vis.plot_legend(win='auc', name=version, y=auc)
            self.vis.plot_legend(win='acc', name=version, y=acc)


def main():
    # tb(Traceback.colour) function should be removed
    import tb
    tb.colour()

    # user code
    ResnetRunner().train_test()


if __name__ == '__main__':
    main()
