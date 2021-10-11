from typing import Dict

import fvcore.nn.weight_init as weight_init
import numpy as np
from detectron2.layers import Conv2d, ShapeSpec
from torch import nn
from torch.nn import functional as F


def build_context_seg_decoder(cfg, input_shape):
    return ContextSegFPNDecoder(cfg, input_shape)


class ContextSegFPNDecoder(nn.Module):
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()

        # fmt: off
        self.in_features = cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
        feature_strides = {k: v.stride for k, v in input_shape.items()}
        feature_channels = {k: v.channels for k, v in input_shape.items()}
        conv_dims = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM  # 128
        # conv_dims             = 256
        self.common_stride = cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE
        norm = cfg.MODEL.SEM_SEG_HEAD.NORM
        # fmt: on

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

    def forward(self, features):
        sum_feat = None

        for i, f in enumerate(self.in_features):
            if i == 0:
                sum_feat = self.scale_heads[i](features[f])
            else:
                sum_feat = sum_feat + self.scale_heads[i](features[f])

        return sum_feat
