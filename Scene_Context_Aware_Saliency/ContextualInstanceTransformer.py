import torch
from torch import nn
from torch.nn import functional as F
import math
import fvcore.nn.weight_init as weight_init


def build_contextual_instance_transformer(cfg):
    return ContextualInstanceTransformer(cfg)


class ContextualInstanceTransformer(nn.Module):
    def __init__(self, cfg, dropout=0.5):
        super().__init__()

        obj_in_dim = 1024
        con_in_dim = 128

        self.ii_dims_inner = 256
        self.ii_dims = 1024

        # Instance-Instance
        self.ii_q_fc = nn.Linear(obj_in_dim, self.ii_dims_inner)
        self.ii_k_fc = nn.Linear(obj_in_dim, self.ii_dims_inner)
        self.ii_v_fc = nn.Linear(obj_in_dim, self.ii_dims)
        weight_init.c2_xavier_fill(self.ii_q_fc)
        weight_init.c2_xavier_fill(self.ii_k_fc)
        weight_init.c2_xavier_fill(self.ii_v_fc)

        self.ii_fc = nn.Linear(self.ii_dims, self.ii_dims)
        # weight_init.c2_xavier_fill(self.ii_fc)
        nn.init.constant_(self.ii_fc.weight, 0)
        nn.init.constant_(self.ii_fc.bias, 0)

        # -----

        self.ic_dims = 128

        # Instance-Context
        self.ic_q_fc = nn.Linear(obj_in_dim, self.ic_dims)
        self.ic_k_conv = nn.Conv2d(con_in_dim, self.ic_dims, kernel_size=1, stride=1, padding=0)
        self.ic_v_conv = nn.Conv2d(con_in_dim, self.ic_dims, kernel_size=1, stride=1, padding=0)
        weight_init.c2_xavier_fill(self.ic_q_fc)
        weight_init.c2_msra_fill(self.ic_k_conv)
        weight_init.c2_msra_fill(self.ic_v_conv)

        self.ic_fc = nn.Linear(self.ic_dims, self.ic_dims)
        # weight_init.c2_msra_fill(self.ic_conv)
        nn.init.constant_(self.ic_fc.weight, 0)
        nn.init.constant_(self.ic_fc.bias, 0)

        # ----------

        self.dropout = nn.Dropout(dropout)

    def forward(self, instance_feat, con_feat):

        num_obj = instance_feat.size()[0]

        # ----------

        # Instance-Instance
        ii_q_x = self.ii_q_fc(instance_feat)
        ii_k_x = self.ii_k_fc(instance_feat)
        ii_v_x = self.ii_v_fc(instance_feat)

        inst_wise_x = self.scaled_dot_product_attention(ii_q_x, ii_k_x, ii_v_x, self.ii_dims_inner, dropout=True)

        del ii_q_x, ii_k_x, ii_v_x
        torch.cuda.empty_cache()

        # Linear
        inst_wise_x = self.ii_fc(inst_wise_x)

        # Add Residual
        inst_wise_x += instance_feat

        # --------------------------------------------------

        # Instance-Context
        ic_q_x = self.ic_q_fc(instance_feat)
        ic_k_x = self.ic_k_conv(con_feat)
        ic_v_x = self.ic_v_conv(con_feat)

        ic_k_x = ic_k_x.view(self.ic_dims, -1)
        ic_v_x = ic_v_x.view(self.ic_dims, -1)

        ic_k_x = torch.transpose(ic_k_x, 0, 1)
        ic_v_x = torch.transpose(ic_v_x, 0, 1)

        con_x = self.scaled_dot_product_attention(ic_q_x, ic_k_x, ic_v_x, self.ic_dims, dropout=True)

        del ic_q_x, ic_k_x, ic_v_x
        torch.cuda.empty_cache()

        # Linear
        con_x = self.ic_fc(con_x)

        # Pool & Repeat for Residual Connection
        pooled_con_x = pool_context_feature(con_feat)
        pooled_con_x = pooled_con_x.repeat(num_obj, 1)

        # Add Residual
        con_x += pooled_con_x

        # --------------------------------------------------

        x = torch.cat([inst_wise_x, con_x], dim=-1)

        return x

    def scaled_dot_product_attention(self, q_feat, k_feat, v_feat, dims, dropout=None):
        # Perform Attention
        scores = torch.matmul(q_feat, k_feat.transpose(-2, -1)) / math.sqrt(dims)

        scores = F.softmax(scores, dim=-1)

        if dropout:
            scores = self.dropout(scores)

        x = torch.matmul(scores, v_feat)

        return x


def pool_context_feature(feature):
    gap = F.avg_pool2d(feature, (feature.size(2), feature.size(3)), stride=(feature.size(2), feature.size(3)))

    x = torch.squeeze(gap, dim=-1)
    x = torch.squeeze(x, dim=-1)

    return x
