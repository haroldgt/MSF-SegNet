from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch_scatter
import torch.nn.functional as F


class Point_Globalfea(nn.Module):
    def __init__(self, fea_compre=None):
        super(Point_Globalfea, self).__init__()
        self.fea_compre = fea_compre

        # point feature compression
        if self.fea_compre is not None:
            self.fea_compression = nn.Sequential(
                nn.Linear(64, self.fea_compre*2),
                nn.ReLU())
            self.pt_fea_dim = self.fea_compre
        else:
            self.pt_fea_dim = self.pool_dim

    def forward(self, shuffled_ind, pt_gfeatures, pt_localfeatures2, pt_inv):
        pt_gfeatures = pt_gfeatures[shuffled_ind, :]
        processed_add_pt_fea = pt_gfeatures + pt_localfeatures2

        pooled_data = torch_scatter.scatter_max(processed_add_pt_fea, pt_inv, dim=0)[0]

        if self.fea_compre*2 > 0 and self.fea_compre*2 < 64:
            processed_pooled_data2 = self.fea_compression(pooled_data)
        else:
            processed_pooled_data2 = pooled_data

        return processed_pooled_data2
