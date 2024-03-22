# -*- coding:utf-8 -*-
# author: Donglin zhu
# title: all net top connection
# @file: Amodel_loader.py

from model.cylinder_Localfea import cylinder_Localfea
from model.Asymm_3d_spconv import Asymm_3d_spconv
from model.cylinder_asym import cylinder_asym
from model.PointNetfeat import PointNetfeat
from model.Point_Globalfea import Point_Globalfea


def build(model_config):
    num_class = model_config['num_class']
    output_shape = model_config['output_shape']
    fea_dim = model_config['fea_dim']
    pointNet_fea_dim = model_config['pointNet_fea_dim']
    num_input_features = model_config['num_input_features']
    base_size = model_config['base_size']

    Localfea_fusion_net = cylinder_Localfea(
                              fea_dim=fea_dim + pointNet_fea_dim,
                              fea_compre=num_input_features)

    asymmetric_3d_spconv_net = Asymm_3d_spconv(
        output_shape=output_shape,
        num_input_features=num_input_features,
        base_size=base_size,
        nclasses=num_class)

    point_feat_net = PointNetfeat(
        global_feat=False,
        feature_transform=False)

    Globalfea_fusion_net = Point_Globalfea(
        fea_compre=num_input_features)

    model = cylinder_asym(
        pointnetfeat_model=point_feat_net,
        cylin_Localfea_model=Localfea_fusion_net,
        point_Globalfea_model=Globalfea_fusion_net,
        asymmetric_spconv_model=asymmetric_3d_spconv_net,
        sparse_shape=output_shape
    )

    return model
