# 这是一个示例 poto net 脚本。

import argparse
import sys
import torch
import numpy as np
import os
import torch.optim as optim
from decimal import Decimal
import time
from config.tools import load_config_parameters
from dataset.dataset import get_label_name
from model import Amodel_loader
from dataset import Anuscenedata_loader
from save_model.Apretrain_loader import load_pretrain_model
from loss import Aloss_loader
from tqdm import tqdm
from val_tools.metric import per_class_iu, fast_hist_crop, var_save_txt, var_save_csv


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':

    """
        level0: parameter
    """
    # 1. parameter file
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-y', '--config_path', default='config/nuScenes.yaml') # config/parameters.yaml
    parser.add_argument('--device', type=int, default=1)
    args = parser.parse_args()
    print('current train file:', ' '.join(sys.argv))  #
    print('parameters file:', args)

    # 2. require parameters
    config_path = args.config_path
    configs = load_config_parameters(config_path)
    torch.backends.cudnn.enabled = True  # option, accelerate training speed when model is fixed.
    torch.backends.cudnn.benchmark = True  # option, accelerate training speed when model is fixed.
    # model options
    model_config = configs['model_params']
    grid_size = model_config['output_shape']
    num_class = model_config['num_class']
    # Dataset options
    dataset_config = configs['dataset_params']
    ignore_label = dataset_config['ignore_label']
    label_mapping_file_path = dataset_config["label_mapping"]
    # train Dataset options
    train_dataloader_config = configs['train_data_loader']
    train_batch_size = train_dataloader_config['batch_size']
    # val Dataset options
    val_dataloader_config = configs['val_data_loader']
    val_batch_size = val_dataloader_config['batch_size']
    # train options
    train_hypers = configs['train_params']
    model_load_path = train_hypers['model_load_path']
    model_save_path = train_hypers['model_save_path']
    csv_save_path = train_hypers['csv_save_path']

    # 3. choice cuda device
    device = args.device
    pytorch_device = torch.device(f"cuda:{device}")

    # 4. After mapping, obtain class labels of the dataset in order, except ignore_label
    label_name = get_label_name(label_mapping_file_path)
    label_number = np.asarray(sorted(list(label_name.keys())))[1:] - 1
    unique_label_str = [label_name[x] for x in label_number + 1]
    print(min(label_number)+1, '-', max(label_number)+1, 'class:', unique_label_str)

    """
        level1: dataset ready
    """
    train_dataset_loader, val_dataset_loader = Anuscenedata_loader.build(dataset_config=dataset_config,
                                                                  train_dataloader_config=train_dataloader_config,
                                                                  val_dataloader_config=val_dataloader_config,
                                                                  grid_size=grid_size)
    """
        level2: model(forward)
    """
    my_model = Amodel_loader.build(model_config)

    """
        level3: training preparation
    """
    # 1. loading pretrained model
    # if os.path.exists(model_save_path):
    #     my_model = load_pretrain_model(model_save_path, my_model)
    # 2. model to gpu
    my_model.to(pytorch_device)
    # 3. define optimizer
    optimizer = optim.Adam(my_model.parameters(), lr=train_hypers["learning_rate"])
    # optimizer = optim.AdamW(my_model.parameters(), lr= train_hypers["learning_rate"])
    # 4. Loss function ready
    cross_entropy, lovasz_softmax = Aloss_loader.build(wce=True,
                                                       lovasz=True,
                                                       num_class=num_class,
                                                       ignore_label=ignore_label)

    """
        level4: train process
    """
    k_epoch = 0
    best_val_mIoU = 0
    my_model.train()
    global_k_iter = 1
    result_txt = np.zeros(4+num_class)
    check_iter = train_hypers['val_every_n_steps']

    # 1. start
    while k_epoch < train_hypers['max_num_epochs']:
        loss_list = []
        # show train progress
        tbar = tqdm(total=len(train_dataset_loader))
        time.sleep(10)
        # 2. input train data iter
        # A*
        for i_iter, (train_grid_pos, train_grid_label, train_pt_xyz, train_pt_grid, train_pt_label, train_pt_fea, train_pt_vfea) in enumerate(train_dataset_loader):
            # 3. input data of a PointCloud to gpu
            train_pt_vfea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in train_pt_vfea] # A*
            train_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in train_pt_fea]
            train_pt_grid_ten = [torch.from_numpy(i).to(pytorch_device) for i in train_pt_grid]
            train_pt_xyz_ten = [torch.from_numpy(i).to(pytorch_device) for i in train_pt_xyz]
            train_grid_label_ten = train_grid_label.type(torch.LongTensor).to(pytorch_device)
            train_batch_size = train_grid_label.shape[0]
            train_pt_size = train_pt_xyz[0].shape[0]
            train_pt_size_ten = torch.tensor(train_pt_size)
            train_pt_size_ten = train_pt_size_ten.type(torch.LongTensor).to(pytorch_device)
            # train_pt_size_ten = torch.from_numpy(train_pt_size).to(pytorch_device)
            # 4. input to model, then calculate loss value between predict and real label
            outputs = my_model(train_pt_fea_ten, train_pt_vfea_ten, train_pt_grid_ten, train_pt_xyz_ten, train_batch_size, train_pt_size_ten)
            loss = lovasz_softmax(torch.nn.functional.softmax(outputs, dim=1), train_grid_label_ten, ignore=0) + cross_entropy(outputs, train_grid_label_ten)
            # 5. backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(my_model.parameters(), 20)  # accelerate model convergence, prevent exploding gradient 
            # 6. optimizer
            optimizer.step()
            # show loss
            loss_list.append(loss.item())
            optimizer.zero_grad()
            tbar.update(1)
            if global_k_iter % check_iter == 0:
                if len(loss_list) > 0:
                    print('  epoch: %d, iter: %5d, loss: %.3f\n' % (k_epoch, i_iter, np.mean(loss_list)))
                else:
                    print('  loss error')

            """
                level5: val
            """
            if (global_k_iter % check_iter == 0) and (k_epoch >= 1):
                my_model.eval()
                hist_list = []
                val_loss_list = []
                result_rows_txt = np.zeros(4+num_class)
                # show val progress
                vbar = tqdm(total=len(val_dataset_loader))
                # 1. in no gradient update state
                with torch.no_grad():
                    # 2. input val data iter
                    for val_i_iter, (_, val_grid_label, val_pt_xyz, val_pt_grid, val_pt_label, val_pt_fea, val_pt_vfea) in enumerate(val_dataset_loader):
                        # 3. input data of a PointCloud to gpu
                        val_pt_vfea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in val_pt_vfea] # A*
                        val_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in val_pt_fea]
                        val_pt_grid_ten = [torch.from_numpy(i).to(pytorch_device) for i in val_pt_grid]
                        val_pt_xyz_ten = [torch.from_numpy(i).to(pytorch_device) for i in val_pt_xyz]
                        val_grid_label_ten = val_grid_label.type(torch.LongTensor).to(pytorch_device)
                        val_batch_size = val_grid_label.shape[0]
                        val_pt_size = val_pt_xyz[0].shape[0]
                        val_pt_size_ten = torch.tensor(val_pt_size)
                        val_pt_size_ten = val_pt_size_ten.type(torch.LongTensor).to(pytorch_device)
                        # 4. input to model
                        predict_outputs = my_model(val_pt_fea_ten, val_pt_vfea_ten, val_pt_grid_ten, val_pt_xyz_ten, val_batch_size, val_pt_size_ten)
                        # predict_outputs express every grid for all class probability, so size 1x20x360x240x16
                        loss = lovasz_softmax(torch.nn.functional.softmax(predict_outputs, dim=1).detach(), val_grid_label_ten, ignore=0) + cross_entropy(predict_outputs.detach(), val_grid_label_ten)
                        # 5. predict result return to cpu
                        predict_outputs = torch.argmax(predict_outputs, dim=1)  # choice max probility as final label for every grid
                        predict_outputs = predict_outputs.cpu().detach().numpy()  # predict_outputs size change to 1x360x240x16
                        # 6. statistic classify circumstance of every point in a PointCloud
                        for count, i_val_grid in enumerate(val_pt_grid):
                            hist_list.append(fast_hist_crop(predict_outputs[count,
                                                                            val_pt_grid[count][:, 0],
                                                                            val_pt_grid[count][:, 1],
                                                                            val_pt_grid[count][:, 2]],
                                                                            val_pt_label[count], label_number))
                        vbar.update(1)
                        val_loss_list.append(loss.detach().cpu().numpy())
                vbar.close()
                # var_save_txt("hist_list", hist_list)
                # show classify result base on IoU index
                print('\n Validation per class iou: ')
                iou = per_class_iu(sum(hist_list))
                ith_class =0
                for class_name, class_iou in zip(unique_label_str, iou):
                    print('      %s : %.2f%%' % (class_name, class_iou * 100))
                    result_rows_txt[ith_class + 4] = Decimal(class_iou).quantize(Decimal("0.0000")) * 100
                    ith_class += 1
                val_mIoU = np.nanmean(iou) * 100
                del val_grid_label, val_pt_grid, val_pt_fea, val_pt_grid_ten
                # save model if performance is improved
                if best_val_mIoU < val_mIoU:
                    best_val_mIoU = val_mIoU
                    torch.save(my_model.state_dict(), model_save_path)
                print('Current val mIoU is %.3f, while the best val mIoU is %.3f.' % (val_mIoU, best_val_mIoU))
                print('Current val loss is %.3f' % (np.mean(val_loss_list)))
                mIoU = round(np.mean(val_loss_list), 3)
                result_rows_txt[0] = k_epoch
                result_rows_txt[1] = Decimal(val_mIoU).quantize(Decimal("0.000"))
                result_rows_txt[2] = Decimal(best_val_mIoU).quantize(Decimal("0.000"))
                result_rows_txt[3] = mIoU
                result_txt = np.vstack((result_txt, result_rows_txt))
                my_model.train()

            global_k_iter += 1
        tbar.close()
        if k_epoch != 0:
            var_save_csv(csv_save_path, result_txt)
        k_epoch += 1


# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
