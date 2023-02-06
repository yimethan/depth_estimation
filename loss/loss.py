import torch
from torch import nn
from loss.losses import *
from loss.utils import *
import numpy as np

class Loss(nn.Module):
    def __init__(self, cfg):
        super(Loss, self).__init__()
        self.down_stride = cfg.down_stride

        self.focal_loss = modified_focal_loss
        self.iou_loss = DIOULoss
        self.l1_loss = F.l1_loss

        self.alpha = cfg.loss_alpha
        self.beta = cfg.loss_beta
        self.gamma = cfg.loss_gamma

    def forward(self, left_pred, right_pred, left_gt, right_gt):

        l_loss = torch.mean(torch.abs(left_gt - left_pred))
        r_loss = torch.mean(torch.abs(right_gt - right_pred))

        return l_loss + r_loss