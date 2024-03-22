# -*- coding:utf-8 -*-
# author: Donglin zhu
# title: Asymm_3d_spconv net
# @file: Asymm_3d_spconv.py

import numpy as np
# import spconv
import spconv.pytorch as spconv
import torch
from torch import nn


def conv3x3(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                             padding=1, bias=False, indice_key=indice_key)


def conv1x3(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(1, 3, 3), stride=stride,
                             padding=(0, 1, 1), bias=False, indice_key=indice_key)


def conv1x1x3(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(1, 1, 3), stride=stride,
                             padding=(0, 0, 1), bias=False, indice_key=indice_key)


def conv1x3x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(1, 3, 1), stride=stride,
                             padding=(0, 1, 0), bias=False, indice_key=indice_key)


def conv3x1x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(3, 1, 1), stride=stride,
                             padding=(1, 0, 0), bias=False, indice_key=indice_key)


def conv3x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(3, 1, 3), stride=stride,
                             padding=(1, 0, 1), bias=False, indice_key=indice_key)


def conv1x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=1, stride=stride,
                             padding=1, bias=False, indice_key=indice_key)


class ResContextBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=(3, 3, 3), stride=1, indice_key=None):
        super(ResContextBlock, self).__init__()
        self.conv1 = conv1x3(in_filters, out_filters, indice_key=indice_key + "bef1")
        self.bn0 = nn.BatchNorm1d(out_filters)
        self.act1 = nn.LeakyReLU()

        self.conv1_2 = conv3x1(out_filters, out_filters, indice_key=indice_key + "bef2")
        # self.conv1_2 = conv1x3(out_filters, out_filters, indice_key=indice_key + "bef")

        self.bn0_2 = nn.BatchNorm1d(out_filters)
        # self.act1_2 = nn.LeakyReLU()

        self.conv2 = conv3x1(in_filters, out_filters, indice_key=indice_key + "bef3")
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(out_filters)

        self.conv3 = conv1x3(out_filters, out_filters, indice_key=indice_key + "bef4")
        # self.conv3 = conv3x1(out_filters, out_filters, indice_key=indice_key + "bef")
        # self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(out_filters)

    #     self.weight_initialization()

    # def weight_initialization(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.BatchNorm1d):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut = shortcut.replace_feature(self.act1(shortcut.features)) # A
        shortcut = shortcut.replace_feature(self.bn0(shortcut.features))

        shortcut = self.conv1_2(shortcut)
        shortcut = shortcut.replace_feature(self.bn0_2(shortcut.features))
        # shortcut = shortcut.replace_feature(self.act1_2(shortcut.features))

        resA = self.conv2(x)
        resA = resA.replace_feature(self.act2(resA.features)) # A
        resA = resA.replace_feature(self.bn1(resA.features))

        resA = self.conv3(resA)
        # resA = resA.replace_feature(self.act3(resA.features))
        resA = resA.replace_feature(self.bn2(resA.features))
        resA = resA.replace_feature(resA.features + shortcut.features)

        return resA


class ResBlock(nn.Module):
    def __init__(self, in_filters, out_filters, dropout_rate, kernel_size=(3, 3, 3), stride=1,
                 pooling=True, drop_out=True, height_pooling=False, indice_key=None):
        super(ResBlock, self).__init__()
        self.pooling = pooling
        self.drop_out = drop_out

        self.conv1 = conv3x1(in_filters, out_filters, indice_key=indice_key + "bef1")
        self.act1 = nn.LeakyReLU()
        self.bn0 = nn.BatchNorm1d(out_filters)

        self.conv1_2 = conv1x3(out_filters, out_filters, indice_key=indice_key + "bef2")
        # self.conv1_2 = conv3x1(out_filters, out_filters, indice_key=indice_key + "bef")
        # self.act1_2 = nn.LeakyReLU()
        self.bn0_2 = nn.BatchNorm1d(out_filters)

        self.conv2 = conv1x3(in_filters, out_filters, indice_key=indice_key + "bef3")
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(out_filters)

        self.conv3 = conv3x1(out_filters, out_filters, indice_key=indice_key + "bef4")
        # self.conv3 = conv1x3(out_filters, out_filters, indice_key=indice_key + "bef")
        # self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(out_filters)
        self.pool = spconv.SparseConv3d(out_filters, out_filters, kernel_size=3, stride=2,
                                                padding=1, indice_key=indice_key, bias=False)

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut = shortcut.replace_feature(self.act1(shortcut.features)) # A
        shortcut = shortcut.replace_feature(self.bn0(shortcut.features))

        shortcut = self.conv1_2(shortcut)
        # shortcut = shortcut.replace_feature(self.act1_2(shortcut.features))
        shortcut = shortcut.replace_feature(self.bn0_2(shortcut.features))

        resA = self.conv2(x)
        resA = resA.replace_feature(self.act2(resA.features)) # A
        resA = resA.replace_feature(self.bn1(resA.features))

        resA = self.conv3(resA)
        # resB = resB.replace_feature(self.act3(resB.features))
        resA = resA.replace_feature(self.bn2(resA.features))

        resA = resA.replace_feature(resA.features + shortcut.features)

        resB = self.pool(resA)

        return resA, resB


class UpBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=(3, 3, 3), indice_key=None):
        super(UpBlock, self).__init__()
        self.trans_dilao1 = conv3x3(in_filters, 2*out_filters, indice_key=indice_key + "newup")
        self.act1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(2*out_filters)

        self.trans_dilao2 = conv3x3(2*out_filters, out_filters, indice_key=indice_key + "newup2")
        self.bn2 = nn.BatchNorm1d(out_filters)
        self.pool = spconv.SparseConv3d(out_filters, out_filters, kernel_size=3, stride=2,
                                                padding=1, indice_key=indice_key, bias=False)

    def forward(self, x):
        resA = self.trans_dilao1(x)
        resA = resA.replace_feature(self.act1(resA.features)) # A
        resA = resA.replace_feature(self.bn1(resA.features))

        resA = self.trans_dilao2(resA)
        resA = resA.replace_feature(self.bn2(resA.features))

        resB = self.pool(resA)

        return resA, resB


class Fusion2line(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=(3, 3, 3), indice_key=None, up_key=None):
        super(Fusion2line, self).__init__()
        # self.drop_out = drop_out
        self.trans_dilao = conv3x3(in_filters, out_filters, indice_key=indice_key + "new_up")
        self.trans_act = nn.LeakyReLU()
        self.trans_bn = nn.BatchNorm1d(out_filters)

        self.conv1 = conv1x3(out_filters, out_filters, indice_key=indice_key + 'up1')
        self.bn1 = nn.BatchNorm1d(out_filters)

        self.conv2 = conv3x1(out_filters, out_filters, indice_key=indice_key + 'up2')
        self.bn2 = nn.BatchNorm1d(out_filters)
        self.up_subm = spconv.SparseInverseConv3d(out_filters, out_filters, kernel_size=3, indice_key=up_key,
                                                  bias=False)

    def forward(self, x, skip):
        x = x.replace_feature(x.features + skip.features)

        upA = self.trans_dilao(x)
        upA = upA.replace_feature(self.trans_act(upA.features)) # A
        upA = upA.replace_feature(self.trans_bn(upA.features))

        upE1 = self.conv1(upA)
        upE1 = upE1.replace_feature(self.bn1(upE1.features))

        upE2 = self.conv2(upA)
        upE2 = upE2.replace_feature(self.bn2(upE2.features))

        upE1 = upE1.replace_feature(upE1.features + upE2.features)

        upE1 = self.up_subm(upE1)

        return upE1


class Fusion3line(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=(3, 3, 3), indice_key=None, up_key=None):
        super(Fusion3line, self).__init__()
        # self.drop_out = drop_out
        self.trans_dilao = conv3x3(in_filters, out_filters, indice_key=indice_key + "new_up")
        self.trans_act = nn.LeakyReLU()
        self.trans_bn = nn.BatchNorm1d(out_filters)

        self.conv1 = conv1x3(out_filters, out_filters, indice_key=indice_key + 'up1')
        self.bn1 = nn.BatchNorm1d(out_filters)

        self.conv2 = conv3x1(out_filters, out_filters, indice_key=indice_key + 'up2')
        self.bn2 = nn.BatchNorm1d(out_filters)
        self.up_subm = spconv.SparseInverseConv3d(out_filters, out_filters, kernel_size=3, indice_key=up_key,
                                                  bias=False)

    def forward(self, x, skip, skip2):
        x = x.replace_feature(x.features + skip.features + skip2.features)

        upA = self.trans_dilao(x)
        upA = upA.replace_feature(self.trans_act(upA.features))
        upA = upA.replace_feature(self.trans_bn(upA.features)) # A

        upE1 = self.conv1(upA)
        upE1 = upE1.replace_feature(self.bn1(upE1.features))

        upE2 = self.conv2(upA)
        upE2 = upE2.replace_feature(self.bn2(upE2.features))

        upE1 = upE1.replace_feature(upE1.features + upE2.features)

        upE1 = self.up_subm(upE1)

        return upE1


class FusionNN(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=(3, 3, 3), indice_key=None):
        super(FusionNN, self).__init__()
        # self.drop_out = drop_out
        self.trans_dilao = conv3x3(in_filters, out_filters, indice_key=indice_key + "finish_up")
        self.trans_act = nn.LeakyReLU()
        self.trans_bn = nn.BatchNorm1d(out_filters)

        self.conv1 = conv1x3(out_filters, out_filters, indice_key=indice_key + 'f_up1')
        self.bn1 = nn.BatchNorm1d(out_filters)

        self.conv2 = conv3x1(out_filters, out_filters, indice_key=indice_key + 'f_up2')
        self.bn2 = nn.BatchNorm1d(out_filters)

    def forward(self, x, skip, skip2):
        x = x.replace_feature(x.features + skip.features + skip2.features)

        upA = self.trans_dilao(x)
        upA = upA.replace_feature(self.trans_act(upA.features)) # A
        upA = upA.replace_feature(self.trans_bn(upA.features))

        upE1 = self.conv1(upA)
        upE1 = upE1.replace_feature(self.bn1(upE1.features))

        upE2 = self.conv2(upA)
        upE2 = upE2.replace_feature(self.bn2(upE2.features))

        upE1 = upE1.replace_feature(upE1.features + upE2.features)

        return upE1


class ReConBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=(3, 3, 3), stride=1, indice_key=None):
        super(ReConBlock, self).__init__()
        self.conv1 = conv3x1x1(in_filters, out_filters, indice_key=indice_key + "bef1")
        self.bn0 = nn.BatchNorm1d(out_filters)
        self.act1 = nn.Sigmoid()

        self.conv1_2 = conv1x3x1(in_filters, out_filters, indice_key=indice_key + "bef2")
        self.bn0_2 = nn.BatchNorm1d(out_filters)
        self.act1_2 = nn.Sigmoid()

        self.conv1_3 = conv1x1x3(in_filters, out_filters, indice_key=indice_key + "bef3")
        self.bn0_3 = nn.BatchNorm1d(out_filters)
        self.act1_3 = nn.Sigmoid()

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut = shortcut.replace_feature(self.act1(shortcut.features))
        shortcut = shortcut.replace_feature(self.bn0(shortcut.features))

        shortcut2 = self.conv1_2(x)
        shortcut2 = shortcut2.replace_feature(self.act1_2(shortcut2.features))
        shortcut2 = shortcut2.replace_feature(self.bn0_2(shortcut2.features))

        shortcut3 = self.conv1_3(x)
        shortcut3 = shortcut3.replace_feature(self.act1_3(shortcut3.features))
        shortcut3 = shortcut.replace_feature(self.bn0_3(shortcut3.features))

        if shortcut.features.shape[1] != x.features.shape[1]:
            shortcut = shortcut.replace_feature(shortcut.features + shortcut2.features + shortcut3.features)
        else :
            shortcut = shortcut.replace_feature(shortcut.features * x.features)

        return shortcut


class Asymm_3d_spconv(nn.Module):
    def __init__(self,
                 output_shape,
                 num_input_features=32,
                 nclasses=20, n_height=32, strict=False, base_size=16):
        super(Asymm_3d_spconv, self).__init__()
        self.nclasses = nclasses
        self.nheight = n_height
        self.strict = False

        sparse_shape = np.array(output_shape)
        # sparse_shape[0] = 11
        print('output_shape:', sparse_shape)
        self.sparse_shape = sparse_shape

        self.ResContextBlock = ResContextBlock(num_input_features, base_size, indice_key="pre")
        self.ReConBlock = ReConBlock(2 * num_input_features, base_size, indice_key="reconStart")

        self.resBlock1 = ResBlock(base_size, 2 * base_size, 0.2, indice_key="down1")
        self.resBlock2 = ResBlock(2 * base_size, 4 * base_size, 0.2, indice_key="down2")
        self.resBlock3 = ResBlock(4 * base_size, 8 * base_size, 0.2, indice_key="down3")
        # self.resBlock4 = ResBlock(8 * base_size, 16 * base_size, 0.2, indice_key="down4")
        
        self.upBlock1 = UpBlock(base_size, 2 * base_size, 0.2, indice_key="down1")
        self.upBlock2 = UpBlock(2 * base_size, 4 * base_size, 0.2, indice_key="down2")
        self.upBlock3 = UpBlock(4 * base_size, 8 * base_size, 0.2, indice_key="down3")
        # self.upBlock4 = UpBlock(8 * base_size, 16 * base_size, 0.2, indice_key="down4")

        # self.fusion2line1 = Fusion2line(16 * base_size, 16 * base_size, indice_key="up1", up_key="down4")
        # self.fusion3line2 = Fusion3line(16 * base_size, 8 * base_size, indice_key="up2", up_key="down3")
        # self.fusion3line3 = Fusion3line(8 * base_size, 4 *base_size, indice_key="up3", up_key="down2")
        # self.fusion3line4 = Fusion3line(4 *base_size, 2 *base_size, indice_key="up4", up_key="down1")
        # self.fusionNN = FusionNN(2 *base_size, base_size, indice_key="up5")
        self.fusion2line1 = Fusion2line(8 * base_size, 8 * base_size, indice_key="up1", up_key="down3")
        self.fusion3line2 = Fusion3line(8 * base_size, 4 * base_size, indice_key="up2", up_key="down2")
        self.fusion3line3 = Fusion3line(4 * base_size, 2 *base_size, indice_key="up3", up_key="down1")
        # self.fusion3line4 = Fusion3line(2 *base_size, base_size, indice_key="up4", up_key="down1")
        self.fusionNN = FusionNN(2 *base_size, base_size, indice_key="up4")

        self.ReconNet = ReConBlock(base_size, base_size, indice_key="reconEnd")

        self.logits = spconv.SubMConv3d(2* base_size, nclasses, indice_key="logit", kernel_size=3, stride=1, padding=1,
                                        bias=True)

    def forward(self, point_features, voxel_features, coors, batch_size):
        # x = x.contiguous()
        coors = coors.int()
        # import pdb
        # pdb.set_trace()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape, batch_size)
        ret2 = spconv.SparseConvTensor(point_features, coors, self.sparse_shape, batch_size)

        ret = self.ResContextBlock(ret)
        ret2 = self.ReConBlock(ret2)

        upl1a, upl1b = self.resBlock1(ret)
        upg1a, upg1b = self.upBlock1(ret2)
        upl2a, upl2b = self.resBlock2(upl1b)
        upg2a, upg2b = self.upBlock2(upg1b)
        upl3a, upl3b = self.resBlock3(upl2b)
        upg3a, upg3b = self.upBlock3(upg2b)
        # upl4a, upl4b = self.resBlock4(upl3b)
        # upg4a, upg4b = self.upBlock4(upg3b)

        # up1e = self.fusion2line1(upl4b, upg4b)
        # up2e = self.fusion3line2(up1e, upl4a, upg4a)
        # up3e = self.fusion3line3(up2e, upl3a, upg3a)
        # up4e = self.fusion3line4(up3e, upl2a, upg2a)
        up1e = self.fusion2line1(upl3b, upg3b)
        up2e = self.fusion3line2(up1e, upl3a, upg3a)
        up3e = self.fusion3line3(up2e, upl2a, upg2a)
        # up4e = self.fusion3line4(up3e, upl1a, upg1a)

        upe = self.fusionNN(up3e, upl1a, upg1a)
        up0e = self.ReconNet(upe)

        up0e = up0e.replace_feature(torch.cat((up0e.features, upe.features), 1))

        logits = self.logits(up0e)
        y = logits.dense()

        return y