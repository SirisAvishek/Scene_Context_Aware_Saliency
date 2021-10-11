# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from ContextualInstanceTransformer import build_contextual_instance_transformer


def build_saliency_box_head(cfg):
    return SaliencyFastRCNNConvFCHead(cfg)


class SaliencyFastRCNNConvFCHead(nn.Module):
    """
    A head with several 3x3 conv layers (each followed by norm & relu) and
    several fc layers (each followed by relu).
    """

    def __init__(self, cfg):
        """
        The following attributes are parsed from config:
            num_conv, num_fc: the number of conv/fc layers
            conv_dim/fc_dim: the dimension of the conv/fc layers
            norm: normalization for the conv layers
        """
        super().__init__()

        num_fc = 3
        fc_dim = cfg.MODEL.ROI_BOX_HEAD.FC_DIM  # 1024

        assert num_fc > 0

        # Number of RoIs per Image
        self.batch_size_per_image = cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE  # 512

        self._output_size = (256, 7, 7)
        self.mask_size = 7

        # ----------

        self.fcs = []
        for k in range(num_fc):
            fc = nn.Linear(np.prod(self._output_size), fc_dim)
            self.add_module("fc{}".format(k + 1), fc)
            self.fcs.append(fc)
            self._output_size = fc_dim
            if k == 0:
                self._output_size = 1152

        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer)

        # ----------

        self.contextual_instance_transformer = build_contextual_instance_transformer(cfg)

        self.out_size = fc_dim

    def forward(self, x, x_batch_index, scene_context_feat):

        if len(self.fcs):
            if x.dim() > 2:
                x = torch.flatten(x, start_dim=1)

            for l_idx, layer in enumerate(self.fcs):

                if l_idx == 1:
                    # Apply Attention
                    x = self.perform_attention(x, scene_context_feat)

                x = F.relu(layer(x))

        return x

    def perform_attention(self, feat_x, con_x):

        # Each RoI Features (from all Image Batch) are currently in the same dimension/channel
        # i.e. [0, 0, 0, 0, 1, 1, 1, 1] - 0=1st img, 1=2nd img
        # So, split the features into separate features corresponding to each image in batch
        # Perform attention for each image features separately
        # Then, Concat back the outputs
        if self.training:
            b_feat_x = torch.split(feat_x, self.batch_size_per_image)

            b_con_x = torch.split(con_x, 1)

            batch_size = len(b_feat_x)

            # List of output for each image in batch
            b_x_list = []
            for b in range(batch_size):
                b_feat = b_feat_x[b]
                b_con = b_con_x[b]

                b_x = self.contextual_instance_transformer(b_feat, b_con)

                b_x_list.append(b_x)

            final_x = torch.cat(b_x_list, dim=0)

            return final_x

        else:
            # Apply Transformer
            final_x = self.contextual_instance_transformer(feat_x, con_x)

            return final_x

    @property
    def output_size(self):
        return self.out_size
