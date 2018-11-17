# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       parser
   Project Name:    octNet_baseline_debug_V10
   Author :         康
   Date:            2018/9/19
-------------------------------------------------
   Change Activity:
                   2018/9/19:
-------------------------------------------------
"""
import argparse

class MyParser(object):
    def __init__(self):
        parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
        parser.add_argument('--arch', '-a', metavar='ARCH', default='',
                            # choices=['simpleAE'],
                            help='model architecture')
        parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                            help='number of data loading workers (default: 4)')
        # useful args
        parser.add_argument('--version', default='v99_01_debug',
                            help='the version of different method/setting/parameters etc.')
        parser.add_argument('--vis-freq', default=15, type=int,
                            help='data sent frequency to visdom server')
        parser.add_argument('--print-freq', '-p', default=60, type=int,
                            metavar='N', help='print frequency (default: 10)')
        # dataset
        parser.add_argument('--data', metavar='DIR', default='/root/dataset/OCT2017',
                            help='path to dataset')
        parser.add_argument('--mean-0', default=0.5, type=float)
        parser.add_argument('--mean-1', default=0.5, type=float)
        parser.add_argument('--mean-2', default=0.5, type=float)
        parser.add_argument('--std-0', default=0.5, type=float)
        parser.add_argument('--std-1', default=0.5, type=float)
        parser.add_argument('--std-2', default=0.5, type=float)

        # retrain
        parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                            help='manual epoch number (useful on restarts)')
        parser.add_argument('--resume', default='', type=str, metavar='PATH',
                            help='path to latest checkpoint (default: none)')
        parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                            help='use pre-trained model')
        # model hyper-parameters
        parser.add_argument('--epochs', default=90, type=int, metavar='N',
                            help='number of total epochs to run')
        parser.add_argument('--batch-size', default=16, type=int,
                            metavar='N', help='mini-batch size (default: 256)')
        parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                            metavar='LR', help='initial learning rate')
        parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                            help='momentum')
        parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                            metavar='W', help='weight decay (default: 1e-4)')
        # visdom
        parser.add_argument('--port', default=31430, type=int,
                            help='visdom server port')
        parser.add_argument('--server', default='10.19.124.11:22143',
                            help='the server of visdom')
        parser.add_argument('--env', default='main',
                            help='the env of Visdom, when do multi experiments, it is very useful')
        parser.add_argument('--img-num', default=5, type=int,
                            help='the num of test img in visdom')
        # others
        parser.add_argument('--classes', default=2, type=int,
                            help='number of classes')
        parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                            help='evaluate model on validation set')

        parser.add_argument('--reverse', action='store_true',
                            help='reverse the train and test dataset')
        parser.add_argument('--seed', default=None, type=int,
                            help='seed for initializing training. ')
        parser.add_argument('--gpu', default=None, type=int,
                            help='GPU id to use.')
        # parser.add_argument('--world-size', default=1, type=int,
        #                     help='number of distributed processes')
        # parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
        #                     help='url used to set up distributed training')
        # parser.add_argument('--dist-backend', default='gloo', type=str,
        #                     help='distributed backend')

        # uspnet, v13
        parser.add_argument('--tau', default=0.5, type=float,
                            help='tau of reconstruct loss in sparse lstm')
        parser.add_argument('--lambd', default=0.5, type=float)

        self.parser = parser


def main():
    pass


if __name__ == '__main__':
    main()
