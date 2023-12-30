# -*- coding:utf-8 -*-
# author: Donglin zhu
# title: all loss functions loader
# @file: Aloss_loader.py

import torch
from loss.lovasz_losses import lovasz_softmax


def build(wce=True, lovasz=True, num_class=20, ignore_label=0):

    cross_entropy = torch.nn.CrossEntropyLoss(ignore_index=ignore_label)
    # cross_entropy = torch.nn.SmoothL1Loss()

    if wce and lovasz:
        return cross_entropy, lovasz_softmax
    elif wce and not lovasz:
        return wce
    elif not wce and lovasz:
        return lovasz_softmax
    else:
        raise NotImplementedError
