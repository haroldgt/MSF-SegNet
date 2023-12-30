# -*- coding:utf-8 -*-
# author: Donglin zhu
# title: iter data generator
# @file: Adata_loader.py

import torch
from dataset.dataset import SemKITTI_sk
from dataset.dataset import cylinder_dataset
from dataset.dataset import collate_fn_BEV


def build(dataset_config,
          test_dataloader_config,
          grid_size=[480, 360, 32]):
    label_mapping_file_path = dataset_config["label_mapping"]
    fixed_volume_space = dataset_config['fixed_volume_space']
    max_volume_space = dataset_config['max_volume_space']
    min_volume_space = dataset_config['min_volume_space']
    ignore_label = dataset_config["ignore_label"]
    data_path = test_dataloader_config["data_path"]
    test_imageset = test_dataloader_config["imageset"]
    test_ref = test_dataloader_config["return_ref"]

    # 1. obtain original PointCloud data
    test_dataset_ptcloud = SemKITTI_sk(data_path=data_path,
                                       imageset=test_imageset,
                                       return_ref=test_ref,
                                       label_mapping=label_mapping_file_path)

    # 2. voxel for a PointCloud data to cylinder coordinate
    test_cylinder_dataset = cylinder_dataset(
        in_dataset=test_dataset_ptcloud,
        grid_size=grid_size,
        fixed_volume_space=fixed_volume_space,
        max_volume_space=max_volume_space,
        min_volume_space=min_volume_space,
        ignore_label=ignore_label,
    )

    # 3. data loading
    test_dataset_loader = torch.utils.data.DataLoader(dataset=test_cylinder_dataset,
                                                      batch_size=test_dataloader_config["batch_size"],
                                                      collate_fn=collate_fn_BEV,
                                                      shuffle=test_dataloader_config["shuffle"],
                                                      num_workers=test_dataloader_config["num_workers"],
                                                      pin_memory=test_dataloader_config["pin_memory"],
                                                      persistent_workers=test_dataloader_config["persistent_workers"])

    return test_dataset_loader
