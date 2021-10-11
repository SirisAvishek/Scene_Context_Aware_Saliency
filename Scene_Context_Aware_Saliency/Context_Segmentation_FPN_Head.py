import fvcore.nn.weight_init as weight_init
from detectron2.layers import Conv2d
from torch import nn
from torch.nn import functional as F


def build_context_seg_head(cfg):
    return ContextSegFPNHead(cfg)


class ContextSegFPNHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # fmt: off
        self.ignore_value = cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE  # 255
        conv_dims = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM  # 128
        self.common_stride = cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE
        # self.loss_weight      = cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT
        self.loss_weight = 1.0
        # fmt: on

        instance_classes = 80 + 1
        stuff_classes = 53 + 1

        self.loss_name_instance = "loss_instance_context_seg"
        self.loss_name_stuff = "loss_stuff_context_seg"

        # ----------

        self.instance_conv = nn.Conv2d(conv_dims, conv_dims, kernel_size=3, stride=1, padding=1)
        self.stuff_conv = nn.Conv2d(conv_dims, conv_dims, kernel_size=3, stride=1, padding=1)
        weight_init.c2_msra_fill(self.instance_conv)
        weight_init.c2_msra_fill(self.stuff_conv)

        # ----------

        self.predictor_instance = Conv2d(conv_dims, instance_classes, kernel_size=1, stride=1, padding=0)
        self.predictor_stuff = Conv2d(conv_dims, stuff_classes, kernel_size=1, stride=1, padding=0)
        weight_init.c2_msra_fill(self.predictor_instance)
        weight_init.c2_msra_fill(self.predictor_stuff)

    def forward(self, features, target_instance=None, target_stuff=None):
        inst_x = self.instance_conv(features)
        inst_x = F.relu(inst_x)

        stuff_x = self.stuff_conv(features)
        stuff_x = F.relu(stuff_x)

        # ---------- Prediction
        # Instance
        inst_x = self.predictor_instance(inst_x)
        inst_x = F.interpolate(inst_x, scale_factor=self.common_stride, mode="bilinear", align_corners=False)

        # Stuff
        stuff_x = self.predictor_stuff(stuff_x)
        stuff_x = F.interpolate(stuff_x, scale_factor=self.common_stride, mode="bilinear", align_corners=False)

        # ----------

        if self.training:
            losses = {}
            losses[self.loss_name_instance] = (
                    F.cross_entropy(inst_x, target_instance.long(), reduction="mean", ignore_index=self.ignore_value)
                    * self.loss_weight
            )
            losses[self.loss_name_stuff] = (
                    F.cross_entropy(stuff_x, target_stuff.long(), reduction="mean", ignore_index=self.ignore_value)
                    * self.loss_weight
            )
            return [], [], losses
        else:
            return inst_x, stuff_x, {}
