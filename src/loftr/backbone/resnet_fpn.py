# from mmcls.models import ResNet

# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# def conv1x1(in_planes, out_planes, stride=1):
#     """1x1 convolution without padding"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


# def conv3x3(in_planes, out_planes, stride=1):
#     """3x3 convolution with padding"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


# class BasicBlock(nn.Module):
#     def __init__(self, in_planes, planes, stride=1):
#         super().__init__()
#         self.conv1 = conv3x3(in_planes, planes, stride)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.relu = nn.ReLU(inplace=True)

#         if stride == 1:
#             self.downsample = None
#         else:
#             self.downsample = nn.Sequential(
#                 conv1x1(in_planes, planes, stride=stride),
#                 nn.BatchNorm2d(planes)
#             )

#     def forward(self, x):
#         y = x
#         y = self.relu(self.bn1(self.conv1(y)))
#         y = self.bn2(self.conv2(y))

#         if self.downsample is not None:
#             x = self.downsample(x)

#         return self.relu(x+y)


# class ResNetFPN_8_2_(nn.Module):
#     """
#     ResNet+FPN, output resolution are 1/8 and 1/2.
#     Each block has 2 layers.
#     """

#     def __init__(self, config):
#         super().__init__()
#         # Config
#         block = BasicBlock
#         initial_dim = config['initial_dim']
#         block_dims = config['block_dims']

#         # Class Variable
#         self.in_planes = initial_dim

#         # Networks
#         self.conv1 = nn.Conv2d(1, initial_dim, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = nn.BatchNorm2d(initial_dim)
#         self.relu = nn.ReLU(inplace=True)

#         self.layer1 = self._make_layer(block, block_dims[0], stride=1)  # 1/2
#         self.layer2 = self._make_layer(block, block_dims[1], stride=2)  # 1/4
#         self.layer3 = self._make_layer(block, block_dims[2], stride=2)  # 1/8

#         # 3. FPN upsample
#         # self.layer3_outconv = conv1x1(block_dims[2], block_dims[2])
#         # self.layer2_outconv = conv1x1(block_dims[1], block_dims[2])
#         # self.layer2_outconv2 = nn.Sequential(
#         #     conv3x3(block_dims[2], block_dims[2]),
#         #     nn.BatchNorm2d(block_dims[2]),
#         #     nn.LeakyReLU(),
#         #     conv3x3(block_dims[2], block_dims[1]),
#         # )
#         # self.layer1_outconv = conv1x1(block_dims[0], block_dims[1])
#         # self.layer1_outconv2 = nn.Sequential(
#         #     conv3x3(block_dims[1], block_dims[1]),
#         #     nn.BatchNorm2d(block_dims[1]),
#         #     nn.LeakyReLU(),
#         #     conv3x3(block_dims[1], block_dims[0]),
#         # )
#         self.layer3_to_layer2 = FeatureAlign_V2(block_dims[1], block_dims[2])
#         self.layer2_outconv2 = nn.Sequential(
#             conv3x3(block_dims[2], block_dims[2]),
#             nn.BatchNorm2d(block_dims[2]),
#             nn.LeakyReLU(),
#             conv3x3(block_dims[2], block_dims[1]),
#         )
#         # self.layer1_outconv = conv1x1(block_dims[0], block_dims[1])
#         self.layer2_to_layer1 = FeatureAlign_V2(block_dims[0], block_dims[1])
#         self.layer1_outconv2 = nn.Sequential(
#             conv3x3(block_dims[1], block_dims[1]),
#             nn.BatchNorm2d(block_dims[1]),
#             nn.LeakyReLU(),
#             conv3x3(block_dims[1], block_dims[0]),
#         )

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#     def _make_layer(self, block, dim, stride=1):
#         layer1 = block(self.in_planes, dim, stride=stride)
#         layer2 = block(dim, dim, stride=1)
#         layers = (layer1, layer2)

#         self.in_planes = dim
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         # ResNet Backbone
#         x0 = self.relu(self.bn1(self.conv1(x)))
#         x1 = self.layer1(x0)  # 1/2
#         x2 = self.layer2(x1)  # 1/4
#         x3 = self.layer3(x2)  # 1/8

#         # FPN
#         # x3_out = self.layer3_outconv(x3)

#         # x3_out_2x = F.interpolate(x3_out, scale_factor=2., mode='bilinear', align_corners=True)
#         # x2_out = self.layer2_outconv(x2)
#         # x2_out = self.layer2_outconv2(x2_out+x3_out_2x)

#         # x2_out_2x = F.interpolate(x2_out, scale_factor=2., mode='bilinear', align_corners=True)
#         # x1_out = self.layer1_outconv(x1)
#         # x1_out = self.layer1_outconv2(x1_out+x2_out_2x)

#         # return [x3_out, x1_out]

#         x2_out = self.layer3_to_layer2(x2, x3)
#         x2_out = self.layer2_outconv2(x2_out)

#         x1_out = self.layer2_to_layer1(x1, x2_out)
#         x1_out = self.layer1_outconv2(x1_out)

#         return [x3, x1_out]



# class ResNetFPN_16_4(nn.Module):
#     """
#     ResNet+FPN, output resolution are 1/16 and 1/4.
#     Each block has 2 layers.
#     """

#     def __init__(self, config):
#         super().__init__()
#         # Config
#         block = BasicBlock
#         initial_dim = config['initial_dim']
#         block_dims = config['block_dims']

#         # Class Variable
#         self.in_planes = initial_dim

#         # Networks
#         self.conv1 = nn.Conv2d(1, initial_dim, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = nn.BatchNorm2d(initial_dim)
#         self.relu = nn.ReLU(inplace=True)

#         self.layer1 = self._make_layer(block, block_dims[0], stride=1)  # 1/2
#         self.layer2 = self._make_layer(block, block_dims[1], stride=2)  # 1/4
#         self.layer3 = self._make_layer(block, block_dims[2], stride=2)  # 1/8
#         self.layer4 = self._make_layer(block, block_dims[3], stride=2)  # 1/16

#         # 3. FPN upsample
#         self.layer4_outconv = conv1x1(block_dims[3], block_dims[3])
#         self.layer3_outconv = conv1x1(block_dims[2], block_dims[3])
#         self.layer3_outconv2 = nn.Sequential(
#             conv3x3(block_dims[3], block_dims[3]),
#             nn.BatchNorm2d(block_dims[3]),
#             nn.LeakyReLU(),
#             conv3x3(block_dims[3], block_dims[2]),
#         )

#         self.layer2_outconv = conv1x1(block_dims[1], block_dims[2])
#         self.layer2_outconv2 = nn.Sequential(
#             conv3x3(block_dims[2], block_dims[2]),
#             nn.BatchNorm2d(block_dims[2]),
#             nn.LeakyReLU(),
#             conv3x3(block_dims[2], block_dims[1]),
#         )

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#     def _make_layer(self, block, dim, stride=1):
#         layer1 = block(self.in_planes, dim, stride=stride)
#         layer2 = block(dim, dim, stride=1)
#         layers = (layer1, layer2)

#         self.in_planes = dim
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         # ResNet Backbone
#         x0 = self.relu(self.bn1(self.conv1(x)))
#         x1 = self.layer1(x0)  # 1/2
#         x2 = self.layer2(x1)  # 1/4
#         x3 = self.layer3(x2)  # 1/8
#         x4 = self.layer4(x3)  # 1/16

#         # FPN
#         x4_out = self.layer4_outconv(x4)

#         x4_out_2x = F.interpolate(x4_out, scale_factor=2., mode='bilinear', align_corners=True)
#         x3_out = self.layer3_outconv(x3)
#         x3_out = self.layer3_outconv2(x3_out+x4_out_2x)

#         x3_out_2x = F.interpolate(x3_out, scale_factor=2., mode='bilinear', align_corners=True)
#         x2_out = self.layer2_outconv(x2)
#         x2_out = self.layer2_outconv2(x2_out+x3_out_2x)

#         return [x4_out, x2_out]

# class ResNetFPN_8_2(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         # Config
#         block = BasicBlock
#         initial_dim = config['initial_dim']
#         block_dims = config['block_dims']

#         # 3. FPN upsample
#         self.layer3_outconv = FeatureSelectionModule(block_dims[2], block_dims[2])
#         self.layer2_outconv = FeatureSelectionModule(block_dims[1], block_dims[2])

#         # self.layer3_to_layer2 = FeatureAlign_V2(block_dims[1], block_dims[2])
#         self.layer2_outconv2 = nn.Sequential(
#             conv3x3(block_dims[2], block_dims[2]),
#             nn.BatchNorm2d(block_dims[2]),
#             nn.LeakyReLU(),
#             conv3x3(block_dims[2], block_dims[1]),
#         )
#         self.layer1_outconv = FeatureSelectionModule(block_dims[0], block_dims[1])
        
#         # self.layer2_to_layer1 = FeatureAlign_V2(block_dims[0], block_dims[1])
#         self.layer1_outconv2 = nn.Sequential(
#             conv3x3(block_dims[1], block_dims[1]),
#             nn.BatchNorm2d(block_dims[1]),
#             nn.LeakyReLU(),
#             conv3x3(block_dims[1], block_dims[0]),
#         )
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#         # Class Variable
#         self.backbone = ResNet(
#                             depth=18,
#                             in_channels=1,
#                             stem_channels=128,
#                             base_channels=128,
#                             use_maxpool=False,
#                             num_stages=3,
#                             out_indices=(0,1,2),
#                             style='pytorch'
#                             )
        
#         # from mmcv.runner import CheckpointLoader, load_state_dict
#         # checkpoint = CheckpointLoader.load_checkpoint(
#         #         '/mnt/lustre/xietao/mmclassification/log/latest.pth', map_location='cpu')
#         # if 'state_dict' in checkpoint:
#         #     state_dict = checkpoint['state_dict']
#         # else:
#         #     state_dict = checkpoint
        
#         # from collections import OrderedDict
#         # new_ckpt = OrderedDict()
#         # for k, v in state_dict.items():
#         #     if k.startswith('backbone'):
#         #         new_key = k.replace('backbone.', '')
#         #         new_ckpt[new_key] = v
#         # # import pdb; pdb.set_trace()
#         # state_dict = new_ckpt
#         # load_state_dict(self.backbone, state_dict, strict=False)
        
#     def forward(self, x):
#         x1, x2, x3 = self.backbone(x)

#         # FPN
#         # x3_out = self.layer3_outconv(x3)

#         # x3_out_2x = F.interpolate(x3_out, scale_factor=2., mode='bilinear', align_corners=True)
#         # x2_out = self.layer2_outconv(x2)

#         # x2_out = self.layer3_to_layer2(x2, x3)
#         # x2_out = self.layer2_outconv2(x2_out)

#         # x1_out = self.layer2_to_layer1(x1, x2_out)
#         # x1_out = self.layer1_outconv2(x1_out)

#         # return [x3, x1_out]

#         # FPN
#         x3_out = self.layer3_outconv(x3)

#         x3_out_2x = F.interpolate(x3_out, scale_factor=2., mode='bilinear', align_corners=True)
#         x2_out = self.layer2_outconv(x2)
#         x2_out = self.layer2_outconv2(x2_out+x3_out_2x)

#         x2_out_2x = F.interpolate(x2_out, scale_factor=2., mode='bilinear', align_corners=True)
#         x1_out = self.layer1_outconv(x1)
#         x1_out = self.layer1_outconv2(x1_out+x2_out_2x)

#         return [x3_out, x1_out]


# class FeatureSelectionModule(nn.Module):
#     def __init__(self, in_chan, out_chan, norm="GN"):
#         super(FeatureSelectionModule, self).__init__()
#         self.conv_atten = nn.Sequential(
#             nn.Conv2d(in_chan, in_chan, kernel_size=1),
#             nn.BatchNorm2d(in_chan)
#         )
#         self.sigmoid = nn.Sigmoid()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_chan, out_chan, kernel_size=1),
#         )

#     def forward(self, x):
#         atten = self.sigmoid(self.conv_atten(F.avg_pool2d(x, x.size()[2:])))
#         feat = torch.mul(x, atten)
#         x = x + feat
#         feat = self.conv(x)
#         return feat


# class FeatureAlign_V2(nn.Module):  # FaPN full version
#     def __init__(self, in_nc=128, out_nc=128, norm=None):
#         super(FeatureAlign_V2, self).__init__()
#         from dcn_v2 import DCN as dcn_v2
#         self.lateral_conv = FeatureSelectionModule(in_nc, out_nc, norm="")
#         self.offset = nn.Conv2d(out_nc * 2, out_nc, kernel_size=1, stride=1, padding=0, bias=False)
#         self.dcpack_L2 = dcn_v2(out_nc, out_nc, 3, stride=1, padding=1, dilation=1, deformable_groups=8, extra_offset_mask=True)
#         self.relu = nn.ReLU(inplace=True)
  
#     def forward(self, feat_l, feat_s, main_path=None):
#         feat_up = F.interpolate(feat_s, scale_factor=2, mode='bilinear', align_corners=True)
#         feat_arm = self.lateral_conv(feat_l)  # 0~1 * feats
#         offset = self.offset(torch.cat([feat_arm, feat_up], dim=1))  # concat for offset by compute the dif
#         feat_align = self.relu(self.dcpack_L2([feat_up, offset]))  # [feat, offset]
#         return feat_align + feat_arm


import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcls.models import ResNet


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution without padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.conv2 = conv3x3(planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        if stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                conv1x1(in_planes, planes, stride=stride),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        y = x
        y = self.relu(self.bn1(self.conv1(y)))
        y = self.bn2(self.conv2(y))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)


class ResNetFPN_8_2(nn.Module):
    """
    ResNet+FPN, output resolution are 1/8 and 1/2.
    Each block has 2 layers.
    """

    def __init__(self, config):
        super().__init__()
        # Config
        block = BasicBlock
        initial_dim = config['initial_dim']
        block_dims = config['block_dims']

        # Class Variable
        self.in_planes = initial_dim

        # Networks
        self.conv1 = nn.Conv2d(1, initial_dim, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(initial_dim)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, block_dims[0], stride=1)  # 1/2
        self.layer2 = self._make_layer(block, block_dims[1], stride=2)  # 1/4
        self.layer3 = self._make_layer(block, block_dims[2], stride=2)  # 1/8

        # 3. FPN upsample
        self.layer3_outconv = FeatureSelectionModule(block_dims[2], block_dims[2])
        self.layer2_outconv = FeatureSelectionModule(block_dims[1], block_dims[2])

        # self.layer3_to_layer2 = FeatureAlign_V2(block_dims[1], block_dims[2])
        self.layer2_outconv2 = nn.Sequential(
            conv3x3(block_dims[2], block_dims[2]),
            nn.BatchNorm2d(block_dims[2]),
            nn.LeakyReLU(),
            conv3x3(block_dims[2], block_dims[1]),
        )
        self.layer1_outconv = FeatureSelectionModule(block_dims[0], block_dims[1])
        
        # self.layer2_to_layer1 = FeatureAlign_V2(block_dims[0], block_dims[1])
        self.layer1_outconv2 = nn.Sequential(
            conv3x3(block_dims[1], block_dims[1]),
            nn.BatchNorm2d(block_dims[1]),
            nn.LeakyReLU(),
            conv3x3(block_dims[1], block_dims[0]),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, dim, stride=1):
        layer1 = block(self.in_planes, dim, stride=stride)
        layer2 = block(dim, dim, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        # ResNet Backbone
        x0 = self.relu(self.bn1(self.conv1(x)))
        x1 = self.layer1(x0)  # 1/2
        x2 = self.layer2(x1)  # 1/4
        x3 = self.layer3(x2)  # 1/8

        # FPN
        x3_out = self.layer3_outconv(x3)

        x3_out_2x = F.interpolate(x3_out, scale_factor=2., mode='bilinear', align_corners=True)
        x2_out = self.layer2_outconv(x2)
        x2_out = self.layer2_outconv2(x2_out+x3_out_2x)

        x2_out_2x = F.interpolate(x2_out, scale_factor=2., mode='bilinear', align_corners=True)
        x1_out = self.layer1_outconv(x1)
        x1_out = self.layer1_outconv2(x1_out+x2_out_2x)

        return [x3_out, x1_out]
        


class ResNetFPN_16_4(nn.Module):
    """
    ResNet+FPN, output resolution are 1/16 and 1/4.
    Each block has 2 layers.
    """

    def __init__(self, config):
        super().__init__()
        # Config
        block = BasicBlock
        initial_dim = config['initial_dim']
        block_dims = config['block_dims']

        # Class Variable
        self.in_planes = initial_dim

        # Networks
        self.conv1 = nn.Conv2d(1, initial_dim, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(initial_dim)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, block_dims[0], stride=1)  # 1/2
        self.layer2 = self._make_layer(block, block_dims[1], stride=2)  # 1/4
        self.layer3 = self._make_layer(block, block_dims[2], stride=2)  # 1/8
        self.layer4 = self._make_layer(block, block_dims[3], stride=2)  # 1/16

        # 3. FPN upsample
        self.layer4_outconv = conv1x1(block_dims[3], block_dims[3])
        self.layer3_outconv = conv1x1(block_dims[2], block_dims[3])
        self.layer3_outconv2 = nn.Sequential(
            conv3x3(block_dims[3], block_dims[3]),
            nn.BatchNorm2d(block_dims[3]),
            nn.LeakyReLU(),
            conv3x3(block_dims[3], block_dims[2]),
        )

        self.layer2_outconv = conv1x1(block_dims[1], block_dims[2])
        self.layer2_outconv2 = nn.Sequential(
            conv3x3(block_dims[2], block_dims[2]),
            nn.BatchNorm2d(block_dims[2]),
            nn.LeakyReLU(),
            conv3x3(block_dims[2], block_dims[1]),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, dim, stride=1):
        layer1 = block(self.in_planes, dim, stride=stride)
        layer2 = block(dim, dim, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        # ResNet Backbone
        x0 = self.relu(self.bn1(self.conv1(x)))
        x1 = self.layer1(x0)  # 1/2
        x2 = self.layer2(x1)  # 1/4
        x3 = self.layer3(x2)  # 1/8
        x4 = self.layer4(x3)  # 1/16

        # FPN
        x4_out = self.layer4_outconv(x4)

        x4_out_2x = F.interpolate(x4_out, scale_factor=2., mode='bilinear', align_corners=True)
        x3_out = self.layer3_outconv(x3)
        x3_out = self.layer3_outconv2(x3_out+x4_out_2x)

        x3_out_2x = F.interpolate(x3_out, scale_factor=2., mode='bilinear', align_corners=True)
        x2_out = self.layer2_outconv(x2)
        x2_out = self.layer2_outconv2(x2_out+x3_out_2x)

        return [x4_out, x2_out]

class ResNetFPN_8_2_xt(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Config
        block = BasicBlock
        initial_dim = config['initial_dim']
        block_dims = config['block_dims']

        # 3. FPN upsample
        self.layer3_outconv = FeatureSelectionModule(block_dims[2], block_dims[2])
        self.layer2_outconv = FeatureSelectionModule(block_dims[1], block_dims[2])

        # self.layer3_to_layer2 = FeatureAlign_V2(block_dims[1], block_dims[2])
        self.layer2_outconv2 = nn.Sequential(
            conv3x3(block_dims[2], block_dims[2]),
            nn.BatchNorm2d(block_dims[2]),
            nn.LeakyReLU(),
            conv3x3(block_dims[2], block_dims[1]),
        )
        self.layer1_outconv = FeatureSelectionModule(block_dims[0], block_dims[1])
        
        # self.layer2_to_layer1 = FeatureAlign_V2(block_dims[0], block_dims[1])
        self.layer1_outconv2 = nn.Sequential(
            conv3x3(block_dims[1], block_dims[1]),
            nn.BatchNorm2d(block_dims[1]),
            nn.LeakyReLU(),
            conv3x3(block_dims[1], block_dims[0]),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # Class Variable
        self.backbone = ResNet(
                            depth=18,
                            in_channels=1,
                            stem_channels=128,
                            base_channels=128,
                            use_maxpool=False,
                            num_stages=3,
                            out_indices=(0,1,2),
                            style='pytorch'
                            )
        from mmcv.runner import CheckpointLoader, load_state_dict
        checkpoint = CheckpointLoader.load_checkpoint(
                '/mnt/lustre/xietao/mmclassification/log/latest.pth', map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        from collections import OrderedDict
        new_ckpt = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('backbone'):
                new_key = k.replace('backbone.', '')
                new_ckpt[new_key] = v
        # import pdb; pdb.set_trace()
        state_dict = new_ckpt
        load_state_dict(self.backbone, state_dict, strict=False)
        
    def forward(self, x):
        x1, x2, x3 = self.backbone(x)

        # FPN
        # x3_out = self.layer3_outconv(x3)

        # x3_out_2x = F.interpolate(x3_out, scale_factor=2., mode='bilinear', align_corners=True)
        # x2_out = self.layer2_outconv(x2)

        # x2_out = self.layer3_to_layer2(x2, x3)
        # x2_out = self.layer2_outconv2(x2_out)

        # x1_out = self.layer2_to_layer1(x1, x2_out)
        # x1_out = self.layer1_outconv2(x1_out)

        # return [x3, x1_out]

        # FPN
        x3_out = self.layer3_outconv(x3)

        x3_out_2x = F.interpolate(x3_out, scale_factor=2., mode='bilinear', align_corners=True)
        x2_out = self.layer2_outconv(x2)
        x2_out = self.layer2_outconv2(x2_out+x3_out_2x)

        x2_out_2x = F.interpolate(x2_out, scale_factor=2., mode='bilinear', align_corners=True)
        x1_out = self.layer1_outconv(x1)
        x1_out = self.layer1_outconv2(x1_out+x2_out_2x)

        return [x3_out, x1_out]


class FeatureSelectionModule(nn.Module):
    def __init__(self, in_chan, out_chan, norm="GN"):
        super(FeatureSelectionModule, self).__init__()
        self.conv_atten = nn.Sequential(
            nn.Conv2d(in_chan, in_chan, kernel_size=1),
            nn.BatchNorm2d(in_chan)
        )
        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=1),
        )

    def forward(self, x):
        atten = self.sigmoid(self.conv_atten(F.avg_pool2d(x, x.size()[2:])))
        feat = torch.mul(x, atten)
        x = x + feat
        feat = self.conv(x)
        return feat


class FeatureAlign_V2(nn.Module):  # FaPN full version
    def __init__(self, in_nc=128, out_nc=128, norm=None):
        super(FeatureAlign_V2, self).__init__()
        from dcn_v2 import DCN as dcn_v2
        self.lateral_conv = FeatureSelectionModule(in_nc, out_nc, norm="")
        self.offset = nn.Conv2d(out_nc * 2, out_nc, kernel_size=1, stride=1, padding=0, bias=False)
        self.dcpack_L2 = dcn_v2(out_nc, out_nc, 3, stride=1, padding=1, dilation=1, deformable_groups=8, extra_offset_mask=True)
        self.relu = nn.ReLU(inplace=True)
  
    def forward(self, feat_l, feat_s, main_path=None):
        feat_up = F.interpolate(feat_s, scale_factor=2, mode='bilinear', align_corners=True)
        feat_arm = self.lateral_conv(feat_l)  # 0~1 * feats
        offset = self.offset(torch.cat([feat_arm, feat_up], dim=1))  # concat for offset by compute the dif
        feat_align = self.relu(self.dcpack_L2([feat_up, offset]))  # [feat, offset]
        return feat_align + feat_arm

