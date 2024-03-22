# -*- coding:utf-8 -*-
# author: Donglin zhu
# title: cylinder_asym net
# @file: cylinder_asym.py

from torch import nn


class cylinder_asym(nn.Module):
    def __init__(self,
                 pointnetfeat_model,
                 cylin_Localfea_model,
                 point_Globalfea_model,
                 asymmetric_spconv_model,
                 sparse_shape,
                 ):
        super().__init__()
        self.name = "cylinder_asym"

        self.pointsfeat_generator = pointnetfeat_model

        self.pointVoxel_fea_fusion_generator = cylin_Localfea_model

        self.voxelPoint_fea_fusion_generator = point_Globalfea_model

        self.intercation_3d_fusion_seg = asymmetric_spconv_model

        self.sparse_shape = sparse_shape

    def forward(self, train_pt_fea_ten, train_pt_vfea_ten,train_vox_ten, train_pt_xyz_ten, batch_size, pt_size):
        point_gfeatures, _, _ = self.pointsfeat_generator(train_pt_xyz_ten, train_pt_xyz_ten, pt_size)

        shuffled_ind, point_localfeatures2, unq_inv, pooled_data1, coords = self.pointVoxel_fea_fusion_generator(train_pt_fea_ten, train_pt_vfea_ten, point_gfeatures, train_vox_ten, train_pt_xyz_ten, pt_size) # A*

        pooled_data2 = self.voxelPoint_fea_fusion_generator(shuffled_ind, point_gfeatures, point_localfeatures2, unq_inv)

        spatial_features = self.intercation_3d_fusion_seg(pooled_data2, pooled_data1, coords, batch_size)

        return spatial_features
