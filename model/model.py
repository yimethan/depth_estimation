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
from model.depth import Depth
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

        self.res_block = centernet._make_layer(block, self)

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

        self.block_3d_1 = centernet._make_layer(block_3d, 64, 64, num_block[1], stride=2)
        self.block_3d_2 = centernet._make_layer(block_3d, 64, 64, num_block[1], stride=2)
        self.block_3d_3 = centernet._make_layer(block_3d, 64, 64, num_block[1], stride=2)
        self.block_3d_4 = centernet._make_layer(block_3d, 64, 128, num_block[1], stride=2)

        self.deconv1=nn.ConvTranspose3d(128, 64, 3, 2, 1, 1)
        self.debn1=nn.BatchNorm3d(64)
        self.deconv2 = nn.ConvTranspose3d(64, 64, 3, 2, 1, 1)
        self.debn2 = nn.BatchNorm3d(64)
        self.deconv3 = nn.ConvTranspose3d(64, 64, 3, 2, 1, 1)
        self.debn3 = nn.BatchNorm3d(64)
        self.deconv4 = nn.ConvTranspose3d(64, 32, 3, 2, 1, 1)
        self.debn4 = nn.BatchNorm3d(32)
        
        self.deconv5 = nn.ConvTranspose3d(32, 1, 3, 2, 1, 1)

    def detects(self, head, return_hm=False, th=None):

        b, c, output_h, output_w = head.pred_hm.shape
        head.pred_hm = CenterNet.pool_nms(head.pred_hm)
        scores, index, clses, ys, xs = self.topk_score(head.pred_hm, K=self.topK)

        reg = CenterNet.gather_feature(head.pred_offset, index, use_transform=True)
        reg = reg.reshape(b, self.topK, 2)
        xs = xs.view(b, self.topK, 1) + reg[:, :, 0:1]
        ys = ys.view(b, self.topK, 1) + reg[:, :, 1:2]

        wh = CenterNet.gather_feature(head.pred_wh, index, use_transform=True)
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
        cost_l = self.gcnet.cost_volume(img_l_1, img_r_1, 'left')
        cost_r = self.gcnet.cost_volume(img_l_1, img_r_1, 'right')

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
        _, _, height, width = left_head.shape

        l_zero_img = np.zeros((height, width))
        r_zero_img = np.zeros((height, width))
        l_re = img_l_0.reshape(height, width)
        r_re = img_r_0.reshape(height, width)
        
        for det in left_detects:
            for i in det[0]: # det[0] : batch boxes
                for x in range(0, height):
                    if (x >= i[1]) & (x <= i[3]):
                        for y in range(0, width):
                            if (y >= i[0]) & (y <= i[2]):
                                l_zero_img[x][y] = l_re[x][y]
        
        for det in right_detects:
            for i in det[0]: # det[0] : batch boxes
                for x in range(0, height):
                    if (x >= i[1]) & (x <= i[3]):
                        for y in range(0, width):
                            if (y >= i[0]) & (y <= i[2]):
                                r_zero_img[x][y] = r_re[x][y]

        # TODO: estimate depth to results of centernet
        carimg_l_0 = F.relu(self.bn0(self.conv0(l_zero_img)))
        carimg_r_0 = F.relu(self.bn0(self.conv0(r_zero_img)))

        carleft_imgblock = self.res_block(carimg_l_0)
        carright_imgblock = self.res_block(carimg_r_0)

        carimg_l_1 = self.conv1(carleft_imgblock)
        carimg_r_1 = self.conv1(carright_imgblock)

        # cost volume
        l_car_cost = CenterNet.cost_volume(carimg_l_1, carimg_r_1, 'left')
        r_car_cost = CenterNet.cost_volume(carimg_l_1, carimg_r_1, 'right')

        # TODO: concatenate cost & car_cost (simply concatenate in 'disp' direction)
        fin_cost_l = torch.cat([cost_l, l_car_cost], 2)
        fin_cost_r = torch.cat([cost_r, r_car_cost], 2)

        conv3d_out_l = F.relu(self.bn3d_1(self.conv3d_1(fin_cost_l)))
        conv3d_out_l = F.relu(self.bn3d_2(self.conv3D_2(conv3d_out_l)))

        conv3d_out_r = F.relu(self.bn3d_1(self.conv3d_1(fin_cost_r)))
        conv3d_out_r = F.relu(self.bn3d_2(self.conv3D_2(conv3d_out_r)))

        # TODO: 3d conv
        #left
        lconv3d_block_1 = self.block_3d_1(fin_cost_l)
        lconv3d_21 = F.relu(self.bn3d_3(self.conv3d_3(fin_cost_l)))
        lconv3d_block_2 = self.block_3d_2(lconv3d_21)
        lconv3d_24 = F.relu(self.bn3d_4(self.conv3d_4(lconv3d_21)))
        lconv3d_block_3 = self.block_3d_3(lconv3d_24)
        lconv3d_27 = F.relu(self.bn3d_5(self.conv3d_5(lconv3d_24)))
        lconv3d_block_4 = self.block_3d_4(lconv3d_27)

        # right
        rconv3d_block_1 = self.block_3d_1(fin_cost_r)
        rconv3d_21 = F.relu(self.bn3d_3(self.conv3d_3(fin_cost_r)))
        rconv3d_block_2 = self.block_3d_2(rconv3d_21)
        rconv3d_24 = F.relu(self.bn3d_4(self.conv3d_4(rconv3d_21)))
        rconv3d_block_3 = self.block_3d_3(rconv3d_24)
        rconv3d_27 = F.relu(self.bn3d_5(self.conv3d_5(rconv3d_24)))
        rconv3d_block_4 = self.block_3d_4(rconv3d_27)

        # TODO: 3d deconv
        # left
        ldeconv3d = F.relu(self.debn1(self.deconv1(lconv3d_block_4))+lconv3d_block_3)
        ldeconv3d = F.relu(self.debn2(self.deconv2(ldeconv3d))+lconv3d_block_2)
        ldeconv3d = F.relu(self.debn3(self.deconv3(ldeconv3d))+lconv3d_block_1)
        ldeconv3d = F.relu(self.debn4(self.deconv4(ldeconv3d))+conv3d_out_l)

        ldeconv3d=self.deconv5(ldeconv3d)

        # right
        rdeconv3d = F.relu(self.debn1(self.deconv1(rconv3d_block_4))+rconv3d_block_3)
        rdeconv3d = F.relu(self.debn2(self.deconv2(rdeconv3d))+rconv3d_block_2)
        rdeconv3d = F.relu(self.debn3(self.deconv3(rdeconv3d))+rconv3d_block_1)
        rdeconv3d = F.relu(self.debn4(self.deconv4(rdeconv3d))+conv3d_out_r)

        rdeconv3d=self.deconv5(rdeconv3d)

        # TODO: soft argmin
        prob_l = F.softmax(-ldeconv3d, 1)
        prob_r = F.softmax(-rdeconv3d, 1)

        return prob_l, prob_r