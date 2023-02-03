from loss.utils import map2coords
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .backbone.renset import ResNet
from model.decoder import Decoder
from model.head import Head
from model.fpn import FPN
from model.centernet import CenterNet
from model.depth import Depth, BasicBlock, ThreeDConv
from config.kitti import Config


class Model(nn.Module):

    def __init__(self, cfg, block, block_3d, num_block, h, w, maxdisp, topK=40):
        
        super(Model, self).__init()

        centernet = CenterNet()
        gcnet = Depth()

        self._fpn = cfg.fpn
        self.down_stride = cfg.down_stride
        self.score_th = cfg.score_th
        self.CLASSES_NAME = cfg.CLASSES_NAME

        self.h = h
        self.w = w
        self.maxdisp = int(maxdisp/2)
        self.in_places = 32

        self.topK = topK

        self.backbone = ResNet(cfg.slug)
        if cfg.fpn:
            self.fpn = FPN(self.backbone.outplanes)
        self.upsample = Decoder(self.backbone.outplanes if not cfg.fpn else 2048, cfg.bn_momentum)
        self.head = Head(channel=cfg.head_channel, num_classes=cfg.num_classes)
        
        # depth
        self.conv0 = nn.Conv2d(3, 3, 2, 5, 2, 2)
        self.bn0 = nn.BatchNorm2d(32)

        self.res_block = self._make_layer(block, self)

        self.conv1=nn.Conv2d(32, 32, 3, 1, 1)

        self.conv3d_1=nn.Conv3d(64, 32, 3, 1, 1)
        self.bn3d_1=nn.BatchNorm3d(32)
        self.conv3d_2=nn.Conv3d(32, 32, 3, 1, 1)
        self.bn3d_2=nn.BatchNorm3d(32)

        self.conv3d_3=nn.Conv3d(64,64,3,2,1)
        self.bn3d_3=nn.BatchNorm3d(64)
        self.conv3d_4=nn.Conv3d(64,64,3,2,1)
        self.bn3d_4=nn.BatchNorm3d(64)
        self.conv3d_5=nn.Conv3d(64,64,3,2,1)
        self.bn3d_5=nn.BatchNorm3d(64)

        self.block_3d_1 = self._make_layer(block_3d,64,64,num_block[1],stride=2)
        self.block_3d_2 = self._make_layer(block_3d, 64, 64, num_block[1], stride=2)
        self.block_3d_3 = self._make_layer(block_3d, 64, 64, num_block[1], stride=2)
        self.block_3d_4 = self._make_layer(block_3d, 64, 128, num_block[1], stride=2)

        self.deconv1=nn.ConvTranspose3d(128,64,3,2,1,1)
        self.debn1=nn.BatchNorm3d(64)
        self.deconv2 = nn.ConvTranspose3d(64, 64, 3, 2, 1, 1)
        self.debn2 = nn.BatchNorm3d(64)
        self.deconv3 = nn.ConvTranspose3d(64, 64, 3, 2, 1, 1)
        self.debn3 = nn.BatchNorm3d(64)
        self.deconv4 = nn.ConvTranspose3d(64, 32, 3, 2, 1, 1)
        self.debn4 = nn.BatchNorm3d(32)
        
        self.deconv5 = nn.ConvTranspose3d(32, 1, 3, 2, 1, 1)

    def cost_volume(self,imgl,imgr):

        xx_list = []

        pad_opr1 = nn.ZeroPad2d((0, self.maxdisp, 0, 0))
        xleft = pad_opr1(imgl)

        for d in range(self.maxdisp):  # maxdisp+1 ?
            pad_opr2 = nn.ZeroPad2d((d, self.maxdisp - d, 0, 0))
            xright = pad_opr2(imgr)
            xx_temp = torch.cat((xleft, xright), 1)
            xx_list.append(xx_temp)

        xx = torch.cat(xx_list, 1)
        xx = xx.view(1, self.maxdisp, 64, int(self.height / 2), int(self.width / 2) + self.maxdisp)
        xx0=xx.permute(0,2,1,3,4)
        xx0 = xx0[:, :, :, :, :int(self.width / 2)]
        
        return xx0
    
    def gather_feature(fmap, index, mask=None, use_transform=False):
        if use_transform:
            # change a (N, C, H, W) tenor to (N, HxW, C) shape
            batch, channel = fmap.shape[:2]
            fmap = fmap.view(batch, channel, -1).permute((0, 2, 1)).contiguous()

        dim = fmap.size(-1)
        index = index.unsqueeze(len(index.shape)).expand(*index.shape, dim)
        fmap = fmap.gather(dim=1, index=index)
        if mask is not None:
            # this part is not called in Res18 dcn COCO
            mask = mask.unsqueeze(2).expand_as(fmap)
            fmap = fmap[mask]
            fmap = fmap.reshape(-1, dim)
        return fmap

    def _make_layer(self,block,in_planes,planes,num_block,stride):
        strides=[stride]+[1]*(num_block-1)
        layers=[]
        for step in strides:
            layers.append(block(in_planes,planes,step))
        return nn.Sequential(*layers)

    def detects(self, head, return_hm=False, th=None):

        b, c, output_h, output_w = head.pred_hm.shape
        head.pred_hm = self.pool_nms(head.pred_hm)
        scores, index, clses, ys, xs = self.topk_score(head.pred_hm, K=self.topK)

        reg = self.gather_feature(head.pred_offset, index, use_transform=True)
        reg = reg.reshape(b, self.topK, 2)
        xs = xs.view(b, self.topK, 1) + reg[:, :, 0:1]
        ys = ys.view(b, self.topK, 1) + reg[:, :, 1:2]

        wh = self.gather_feature(head.pred_wh, index, use_transform=True)
        wh = wh.reshape(b, self.topK, 2)

        clses = clses.reshape(b, self.topK, 1).float()
        scores = scores.reshape(b, self.topK, 1)

        half_w, half_h = wh[..., 0:1] / 2, wh[..., 1:2] / 2
        bboxes = torch.cat([xs - half_w, ys - half_h, xs + half_w, ys + half_h], dim=2)

        detects = []
        for batch in range(b):
            mask = scores[batch].gt(self.score_th if th is None else th)

            batch_boxes = bboxes[batch][mask.squeeze(-1), :]
            batch_boxes[:, [0, 2]] *= self.w / output_w
            batch_boxes[:, [1, 3]] *= self.h / output_h

            batch_scores = scores[batch][mask]

            batch_clses = clses[batch][mask]
            batch_clses = [self.CLASSES_NAME[int(cls.item())] for cls in batch_clses]

            detects.append([batch_boxes, batch_scores, batch_clses, head.pred_hm[batch] if return_hm else None])

        return detects

    def pool_nms(self, hm, pool_size=3):
        pad = (pool_size - 1) // 2
        hm_max = F.max_pool2d(hm, pool_size, stride=1, padding=pad)
        keep = (hm_max == hm).float()
        return hm * keep

    def forward(self, left_img, right_img):

        # TODO: estimate depth of left & right image
        img_l_0 = F.relu(self.bn0(self.conv0(left_img)))
        img_r_0 = F.relu(self.bn0(self.conv0(right_img)))

        left_imgblock = self.res_block(img_l_0)
        right_imgblock = self.res_block(img_r_0)

        img_l_1 = self.conv1(left_imgblock)
        img_r_1 = self.conv1(right_imgblock)

        # cost volume
        left_cost = self.cost_volume(img_l_1)
        l_conv3d_out = F.relu(self.bn3d_1(self.conv3d_1(left_cost)))
        l_conv3d_out = F.relu(self.bn3d_2(self.conv3D_2(l_conv3d_out)))

        right_cost = self.cost_volume(img_r_1)
        r_conv3d_out = F.relu(self.bn3d_1(self.conv3d_1(right_cost)))
        r_conv3d_out = F.relu(self.bn3d_2(self.conv3D_2(r_conv3d_out)))

        # TODO: centernet
        left_feats = self.backbone(left_img)
        if self._fpn:
            left_feat = self.fpn(left_feats)
        else:
            left_feat = left_feats[-1]
        left_head = self.head(self.upsample(left_feat)) # pred_hm, pred_wh, pred_offset

        right_feats = self.backbone(right_img)
        if self._fpn:
            right_feat = self.fpn(right_feats)
        else:
            right_feat = right_feats[-1]
        right_head = self.head(self.upsample(right_feat))

        left_detects = self.detects(left_head) # list of [batch_boxes, batch_scores, batch_clses, head.pred_hm[batch] if return_hm else None]
        right_detects = self.detects(right_head)

        # TODO: modify results of centernet (zero array with only vehicle images)
        l_zero_img = np.zeros((self.h, self.w))
        r_zero_img = np.zeros((self.h, self.w))
        

        # TODO: estimate depth to results of centernet

        # TODO: during left + right disparity concatenation, also concatenate small disparity

        # TODO: 3d conv

        # TODO: 3d deconv

        # TODO: generate depth map