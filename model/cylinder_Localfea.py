# -*- coding:utf-8 -*-
# author: Donglin zhu
# title: cylinder_fea net
# @file: cylinder_fea.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import numba as nb
import multiprocessing
import torch_scatter
from val_tools.metric import var_save_txt
from model.PointNetfeat import PointTransformerLayer
import model.pytorch_utils as pt_utils


class cylinder_Localfea(nn.Module):

    def __init__(self, fea_dim=3, fea_compre=None):
        super(cylinder_Localfea, self).__init__()

        # self.toset = nn.Linear(64, 7)
        self.linear_combination = nn.Linear(7, 3) # A*
        # self.att_combination = Att_pooling(4,3) # A*

        self.PPmodel = nn.Sequential(
            nn.BatchNorm1d(fea_dim),

            nn.Linear(fea_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        self.PPmodel2 = nn.Sequential(
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(32, 32)
        )
        self.conv1 = torch.nn.Conv1d(16, 32, 1)
        self.conv2 = torch.nn.Conv1d(32, 64, 1)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(32)

        self.fea_compre = fea_compre

        # voxel feature compression
        if self.fea_compre is not None:
            self.fea_compression = nn.Sequential(
                nn.Linear(32, self.fea_compre),
                nn.ReLU())
            self.pt_fea_dim = self.fea_compre
        else:
            self.pt_fea_dim = self.pool_dim

    def forward(self, pt_fea, pt_Vfea, pt_features, xy_ind, pt_xyz, pt_size):
        cur_dev = pt_fea[0].get_device()

        # concat everything
        cat_pt_ind = []
        for i_batch in range(len(xy_ind)):
            cat_pt_ind.append(F.pad(xy_ind[i_batch], (1, 0), 'constant', value=i_batch))

        pt_fea = torch.cat(pt_fea, dim=0)

        xyz_vn = torch.concat((pt_xyz[0], pt_Vfea[0]), dim=1) # A*
        v_fea = self.linear_combination(xyz_vn) # A*
        # v_fea = self.att_combination(xyz_vn)
        cat_pt_fea = torch.concat((pt_fea, v_fea), dim=1) # A*

        pt_features = torch.max(pt_features, 1, keepdim=True)[0]

        processed_cat_pt_fea = torch.concat((cat_pt_fea, pt_features), dim=1)
        cat_pt_ind = torch.cat(cat_pt_ind, dim=0)
        pt_num = cat_pt_ind.shape[0]

        # shuffle the data
        shuffled_ind = torch.randperm(pt_num, device=cur_dev)
        processed_cat_pt_fea = processed_cat_pt_fea[shuffled_ind, :]
        cat_pt_ind = cat_pt_ind[shuffled_ind, :]

        # unique xy grid index
        unq, unq_inv, unq_cnt = torch.unique(cat_pt_ind, return_inverse=True, return_counts=True, dim=0)
        unq = unq.type(torch.int64)

        # process feature
        processed_cat_pt_fea = self.PPmodel(processed_cat_pt_fea)

        point_localfeatures1 = self.PPmodel2(processed_cat_pt_fea)

        point_localfeatures2 = torch.unsqueeze(processed_cat_pt_fea, dim=0)
        point_localfeatures2 = point_localfeatures2.transpose(2, 1)
        point_localfeatures2 = F.relu(self.bn1(self.conv1(point_localfeatures2)))
        point_localfeatures2 = self.conv2(point_localfeatures2)
        point_localfeatures2 = point_localfeatures2.transpose(2, 1)
        point_localfeatures2 = torch.squeeze(point_localfeatures2)

        pooled_data = torch_scatter.scatter_max(point_localfeatures1, unq_inv, dim=0)[0]

        if self.fea_compre > 0 and self.fea_compre < 32:
            processed_pooled_data1 = self.fea_compression(pooled_data)
        else:
            processed_pooled_data1 = pooled_data

        return shuffled_ind, point_localfeatures2, unq_inv, processed_pooled_data1, unq


class Att_pooling(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.fc = nn.Conv2d(d_in, d_in, (1, 1), bias=False)
        self.mlp = pt_utils.Conv2d(d_in, d_out, kernel_size=(1, 1), bn=True)

    def forward(self, feature_set):
        # input: n*d_in, output: 
        x_in = torch.unsqueeze(feature_set, dim=0)
        x_in = x_in.transpose(2, 1)
        x_in = torch.unsqueeze(x_in, dim=3)

        att_activation = self.fc(x_in)
        att_scores = F.softmax(att_activation, dim=3)
        f_agg = x_in * att_scores
        f_agg = torch.sum(f_agg, dim=3, keepdim=True)
        f_agg = self.mlp(f_agg)

        x_out = torch.squeeze(f_agg, dim=3)
        x_out = x_out.transpose(2, 1)
        x_out = torch.squeeze(x_out, dim=0)

        return x_out
