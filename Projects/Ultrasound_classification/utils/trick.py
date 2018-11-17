# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       trick
   Project Name:    segOCT
   Author :         Kang ZHOU
   Date:            2018/11/6
-------------------------------------------------
   Change Activity:
                   2018/11/6:
-------------------------------------------------
"""
import os
import warnings
import shutil

import torch
import itchat


def adjust_lr(lr, optimizer, epoch, e_freq=30):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // e_freq))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def l1_reg(model):
    l1_loss = 0
    cnt = 0
    for p in model.parameters():
        cnt += 1
        l1_loss += p.abs().sum()
    return l1_loss / (cnt + 0.000001)

def save_ckpt(version, state, is_best, epoch, project):
    v_split_list = version.split('_')
    v_major = v_split_list[0]
    v_minor = v_split_list[1]

    ckpt_dir = os.path.join('/root/workspace/', project, 'checkpoints')
    version_filename = '{}_ckpt.pth.tar'.format(version)
    version_file_path = os.path.join(ckpt_dir, version_filename)
    torch.save(state, version_file_path)
    # if epoch % 10 == 0:
    #     ckpt_file_path = os.path.join(ckpt_dir, '{}_ckpt@{}.pth.tar'.format(version, epoch))
    #     torch.save(state, ckpt_file_path)
    if is_best:
        best_file_path = os.path.join(ckpt_dir, '{}_{}_best@{}.pth.tar'.format(v_major, v_minor, epoch))
        # shutil.copyfile(version_file_path, best_file_path)  maintain the best model
        torch.save(state, best_file_path)

def cuda_visible(gpu_list):
    if gpu_list == None:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        warnings.warn('You should better speicify the gpu id. The default gpu is 0.')
    elif len(gpu_list) == 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(gpu_list[0])
    elif len(gpu_list) == 2:
        os.environ['CUDA_VISIBLE_DEVICES'] = '{},{}'.format(gpu_list[0], gpu_list[1])
    elif len(gpu_list) == 3:
        os.environ['CUDA_VISIBLE_DEVICES'] = '{},{},{}'.format(gpu_list[0], gpu_list[1], gpu_list[2])
    elif len(gpu_list) == 4:
        os.environ['CUDA_VISIBLE_DEVICES'] = '{},{},{},{}'. \
            format(gpu_list[0], gpu_list[1], gpu_list[2], gpu_list[3])
    else:
        raise ValueError('wrong in gpu list')

def print_args(args):
    print('\n', '*' * 30, 'Args', '*' * 30)
    print('Args: \n{}\n'.format(args))

def range_norm(image):
    if image.dim() == 2:
        _min = image.min()
        _range = image.max() - image.min()
        return (image - _min) / _range
    else:
        print('Not implement!')

class WeiXin(object):
    def __init__(self):
        itchat.auto_login(enableCmdQR=True)
        self.users = itchat.search_friends(name='今天你磕盐了吗')

    def send(self, message):
        userName = self.users[0]['UserName']
        itchat.send(message, toUserName=userName)

def main():
    # tb(Traceback.colour) function should be removed
    import tb
    tb.colour()

    # user code


if __name__ == '__main__':
    main()
