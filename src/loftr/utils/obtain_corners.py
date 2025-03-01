import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

INF = 1e9

def border_mask(x, mask=None, board_num=2):
    """
    :param board_num: mask board num
    :param x: raw data
    :param mask: masking outdoor padding
    :return: x after padding
    """
    assert board_num >= 0
    x[:, :board_num, :] = -1
    x[:, -board_num:, :] = -1
    x[:, :, :board_num] = -1
    x[:, :, -board_num:] = -1
    if mask is None:
        return x
    x[~mask] = -1
    return x

class generate_corners(nn.Module):
    def __init__(self, kernel_sizes=[3, 5, 7], paddings=[1, 2, 3]):
        super(generate_corners, self).__init__()
        self.kernels = nn.ParameterList([
            nn.Parameter(torch.ones((sz, sz)).unsqueeze(0).unsqueeze(0), requires_grad=False) for sz in kernel_sizes])
        self.weights = nn.ParameterList([
            nn.Parameter((torch.ones((1, 1)) * (sz * sz)).unsqueeze(0).unsqueeze(0), requires_grad=False) for sz in kernel_sizes])
        self.padding = paddings
        self.bn = nn.ModuleList([nn.BatchNorm2d(1) for i in range(3)])
        self.relu = nn.ReLU()
    
    def conv_layer(self, fmap, weight0, weight1, padding, bn):
        x1 = F.conv2d(fmap, weight0, padding=padding)
        x2 = F.conv2d(fmap, weight1, padding=0)

        att = x2 - x1
        att = bn(att)
        att = self.relu(att)
        return att

    # [B C H W]  [B H W]
    # b_index0_1  i_index_1  j_index_1[B N]
    def generate_seed_forward(self, data, x, mask=None, seed_num_list=None):
        assert seed_num_list is not None, print("Seed_number == 0! Error!")
        mask0 = mask1 = None  # mask is useful in training
        if 'mask0' in data:
            mask0, mask1 = data['mask0'], data['mask1']     # [B H W]
        
        fmap = x.sum(1, keepdim=True)
        x_3 = self.conv_layer(fmap, self.kernels[0], self.weights[0], self.padding[0], self.bn[0])
        x_5 = self.conv_layer(fmap, self.kernels[1], self.weights[1], self.padding[1], self.bn[1])
        x_7 = self.conv_layer(fmap, self.kernels[2], self.weights[2], self.padding[2], self.bn[2])   
        final_att = x_3 + x_5 + x_7
        
        final_att = border_mask(final_att.squeeze(dim=1), mask)
        final_att = final_att.flatten(1)                # [B N]

        B = final_att.shape[0]

        b_index_list = []
        i_index_list = []
        for seed_num in seed_num_list:
            val, i_index = torch.topk(final_att, dim=-1, k=seed_num)
            b_index = torch.arange(B)[:, None].repeat(1, seed_num)
            b_index_list.append(b_index)
            i_index_list.append(i_index)
        return b_index_list, i_index_list
















































