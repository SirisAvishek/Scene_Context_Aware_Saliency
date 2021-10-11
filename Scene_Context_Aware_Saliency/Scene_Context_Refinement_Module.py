import fvcore.nn.weight_init as weight_init
import numpy as np
import torch
from detectron2.layers import Conv2d
from torch import nn
from torch.nn import functional as F


def build_scene_context_refinement_module(cfg):
    return SceneContextRefinementModule(cfg)


class SceneContextRefinementModule(nn.Module):
    """
       Context Refinement Module
    """
    def __init__(self, cfg):
        super().__init__()

        conv_dims = 128

        norm = cfg.MODEL.SEM_SEG_HEAD.NORM

        feature_channels = {'p3': 256, 'p4': 256, 'p5': 256, 'p6': 256}
        feature_strides = {'p3': 8, 'p4': 16, 'p5': 32, 'p6': 64}
        self.in_features = ['p3', 'p4', 'p5']

        self.common_stride = cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE  # 4

        self.scale_heads = []
        for in_feature in self.in_features:
            head_ops = []
            head_length = max(
                1, int(np.log2(feature_strides[in_feature]) - np.log2(self.common_stride))
            )
            for k in range(head_length):
                norm_module = nn.GroupNorm(32, conv_dims) if norm == "GN" else None
                conv = Conv2d(
                    feature_channels[in_feature] if k == 0 else conv_dims,
                    conv_dims,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=not norm,
                    norm=norm_module,
                    activation=F.relu,
                )
                weight_init.c2_msra_fill(conv)
                head_ops.append(conv)
                if feature_strides[in_feature] != self.common_stride:
                    head_ops.append(
                        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
                    )
            self.scale_heads.append(nn.Sequential(*head_ops))
            self.add_module(in_feature, self.scale_heads[-1])

        # ----------

        self.conv_reduce = nn.Conv2d(512, conv_dims, kernel_size=1, stride=1, padding=0)
        weight_init.c2_msra_fill(self.conv_reduce)

        self.crm = CRM(conv_dims)
        self.srm = SRM(conv_dims)

        self.final_conv = nn.Conv2d(conv_dims, conv_dims, kernel_size=1, stride=1, padding=0)
        weight_init.c2_msra_fill(self.final_conv)

    def forward(self, fpn_features, context_feat):
        # FPN Refinement
        f_features = []
        for i, f in enumerate(self.in_features):
            feat = self.scale_heads[i](fpn_features[f])
            f_features.append(feat)

        # ------------------------------

        f_features.append(context_feat)

        multi_feat = torch.cat(f_features, dim=1)

        ft_x = self.conv_reduce(multi_feat)
        ft_x = F.relu(ft_x)

        # ----- Apply Attentions
        # Channel-wise Attention
        ca_x = self.crm(ft_x)

        # Spatial-wise Attention
        sa_x = self.srm(ft_x)

        x = ca_x * sa_x

        x += context_feat

        # ------------------------------

        x = self.final_conv(x)
        x = F.relu(x)

        return x


class CRM(nn.Module):
    """
        Channel-wise Refinement Module
    """
    def __init__(self, dim):
        super().__init__()

        r = 16

        self.conv_1 = nn.Conv2d(dim, dim // r, kernel_size=1, stride=1, padding=0)
        self.conv_2 = nn.Conv2d(dim // r, dim, kernel_size=1, stride=1, padding=0)
        weight_init.c2_msra_fill(self.conv_1)
        weight_init.c2_msra_fill(self.conv_2)

    def forward(self, feature):
        b, c, _, _ = feature.size()

        gap = F.avg_pool2d(feature, (feature.size(2), feature.size(3)), stride=(feature.size(2), feature.size(3)))
        gmp = F.max_pool2d(feature, (feature.size(2), feature.size(3)), stride=(feature.size(2), feature.size(3)))

        # --------------------

        gap = self.conv_1(gap)
        gap = F.relu(gap)
        gap = self.conv_2(gap)

        gmp = self.conv_1(gmp)
        gmp = F.relu(gmp)
        gmp = self.conv_2(gmp)

        x = gap + gmp

        x = torch.sigmoid(x)

        x = feature * x

        return x


class SRM(nn.Module):
    """
        Spatial Refinement Module
    """
    def __init__(self, channels):
        super().__init__()

        kernel_size = 7

        self.conv_1_1 = nn.Conv2d(channels, channels,
                                  kernel_size=(kernel_size, 1),
                                  padding=(kernel_size // 2, 0))
        self.conv_1_2 = nn.Conv2d(channels, channels,
                                  kernel_size=(1, kernel_size),
                                  padding=(0, kernel_size // 2))

        self.conv_2_1 = nn.Conv2d(channels, channels,
                                  kernel_size=(1, kernel_size),
                                  padding=(0, kernel_size // 2))
        self.conv_2_2 = nn.Conv2d(channels, channels,
                                  kernel_size=(kernel_size, 1),
                                  padding=(kernel_size // 2, 0))

        weight_init.c2_msra_fill(self.conv_1_1)
        weight_init.c2_msra_fill(self.conv_1_2)
        weight_init.c2_msra_fill(self.conv_2_1)
        weight_init.c2_msra_fill(self.conv_2_2)

    def forward(self, features):
        x_1 = self.conv_1_1(features)
        x_1 = F.relu(x_1)
        x_1 = self.conv_1_2(x_1)
        x_1 = F.relu(x_1)

        x_2 = self.conv_2_1(features)
        x_2 = F.relu(x_2)
        x_2 = self.conv_2_2(x_2)
        x_2 = F.relu(x_2)

        # ----------

        x = x_1 + x_2

        x = torch.sigmoid(x)

        x = features * x

        return x
