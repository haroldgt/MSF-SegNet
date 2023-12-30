# -*- coding:utf-8 -*-
# author: Donglin zhu
# title: iter data generator
# @file: Adata_loader.py

import torch
from dataset.nuscene_dataset import nuScene_sk
from dataset.nuscene_dataset import cylinder_dataset
from dataset.nuscene_dataset import collate_fn_BEV
from nuscenes import NuScenes


def build(dataset_config,
          train_dataloader_config,
          val_dataloader_config,
          grid_size=[480, 360, 32]):
    label_mapping_file_path = dataset_config["label_mapping"]
    labelData_bits = dataset_config["labelData_bits"]
    fixed_volume_space = dataset_config['fixed_volume_space']
    max_volume_space = dataset_config['max_volume_space']
    min_volume_space = dataset_config['min_volume_space']
    ignore_label = dataset_config["ignore_label"]
    data_path = train_dataloader_config["data_path"]
    train_imageset = train_dataloader_config["imageset"]
    train_ref = train_dataloader_config["return_ref"]
    val_imageset = val_dataloader_config["imageset"]
    val_ref = val_dataloader_config["return_ref"]

    nusc=None
    nusc = NuScenes(version='v1.0-trainval', dataroot=data_path, verbose=True)

    # 1. obtain original PointCloud data
    train_dataset_ptcloud = nuScene_sk(data_path=data_path,
                                        imageset=train_imageset,
                                        return_ref=train_ref,
                                        label_mapping=label_mapping_file_path,
                                        labelData_bits = labelData_bits, 
                                        nusc=nusc)
    val_dataset_ptcloud = nuScene_sk(data_path=data_path,
                                      imageset=val_imageset,
                                      return_ref=val_ref,
                                      label_mapping=label_mapping_file_path,
                                      labelData_bits = labelData_bits, 
                                      nusc=nusc)

    # 2. voxel for a PointCloud data to cylinder coordinate
    train_cylinder_dataset = cylinder_dataset(
        in_dataset=train_dataset_ptcloud,
        grid_size=grid_size,
        flip_aug=True,
        fixed_volume_space=fixed_volume_space,
        max_volume_space=max_volume_space,
        min_volume_space=min_volume_space,
        ignore_label=ignore_label,
        rotate_aug=True,
        scale_aug=True,
        transform_aug=True
    )

    val_cylinder_dataset = cylinder_dataset(
        in_dataset=val_dataset_ptcloud,
        grid_size=grid_size,
        fixed_volume_space=fixed_volume_space,
        max_volume_space=max_volume_space,
        min_volume_space=min_volume_space,
        ignore_label=ignore_label,
    )

    # 3. data loading
    train_dataset_loader = torch.utils.data.DataLoader(dataset=train_cylinder_dataset,
                                                       batch_size=train_dataloader_config["batch_size"],
                                                       collate_fn=collate_fn_BEV,
                                                       shuffle=train_dataloader_config["shuffle"],
                                                       num_workers=train_dataloader_config["num_workers"],
                                                       pin_memory=train_dataloader_config["pin_memory"],
                                                       persistent_workers=train_dataloader_config["persistent_workers"])
    val_dataset_loader = torch.utils.data.DataLoader(dataset=val_cylinder_dataset,
                                                     batch_size=val_dataloader_config["batch_size"],
                                                     collate_fn=collate_fn_BEV,
                                                     shuffle=val_dataloader_config["shuffle"],
                                                     num_workers=val_dataloader_config["num_workers"],
                                                     pin_memory=train_dataloader_config["pin_memory"],
                                                     persistent_workers=train_dataloader_config["persistent_workers"])

    return train_dataset_loader, val_dataset_loader
