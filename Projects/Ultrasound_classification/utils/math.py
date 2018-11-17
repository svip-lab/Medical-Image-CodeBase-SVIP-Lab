# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       math
   Project Name:    octNet
   Author :         康
   Date:            2018/10/14
-------------------------------------------------
   Change Activity:
                   2018/10/14:
-------------------------------------------------
"""
import numpy as np, scipy.stats as st
import torch


def calc_confidence_interval(samples, confidence_value=0.95):
    # samples should be a numpy array
    if type(samples) is list:
        samples = np.asarray(samples)
    assert isinstance(samples, np.ndarray), 'args: samples {} should be np.array'.format(samples)
    # print('Results List:', samples)
    stat_accu = st.t.interval(confidence_value, len(samples)-1, loc=np.mean(samples), scale=st.sem(samples))
    center = (stat_accu[0]+stat_accu[1])/2
    deviation = (stat_accu[1]-stat_accu[0])/2
    return center, deviation

def de_normalize(tensor):
    """
    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
    Returns:
        Tensor: Normalized image.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if tensor.dim() == 3:
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
        # The normalize code -> t.sub_(m).div_(s)
    elif tensor.dim() == 4:
        # (N, C, H, W) => (C, N, H, W)
        tensor = tensor.transpose(0, 1)
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
        tensor = tensor.transpose(0, 1)
    return tensor


def main():
    a = np.random.randn(100)
    b = np.random.randn(1000)

    print(calc_confidence_interval(a), calc_confidence_interval(b))


if __name__ == '__main__':
    main()
