import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from dataset.transform import Transform
from model.detect import DlaNet
from config.config import Config
from tensorboardX import SummaryWriter

class Model(nn.Module):

    def __init__(self):

        print("Creating model...")

        super(Model, self).__init__()

        self.detect_model = DlaNet()
        self.detect_model.load_state_dict(torch.load(Config.centernet_path))

        # freeze detection model's layers
        for param in self.detect_model.parameters():
            param.requires_grad = False

        # gc based
        self.depth_model = DepthBlock(BasicBlock, ThreedConv, [8, 1])

        self.transform = Transform()

    def forward(self, x):
        
        # TODO: detect vehicles in each image
        # (h, w, 3) -> (3, h, w) -> (1, 3, h, w)
        l_inp = x['l_img'].reshape(1, 3, Config.height, Config.width)
        r_inp = x['r_img'].reshape(1, 3, Config.height, Config.width)

        with torch.no_grad():
            left_y = self.detect_model(l_inp)
            right_y = self.detect_model(r_inp)

            hm = left_y['hm'].sigmoid_()
            wh = left_y['wh']
            reg = left_y['reg']

            heads = [hm, wh, reg]
            l_dets = self.ctdet_decode(heads, 40) # K is the number of remaining instances

            hm = right_y['hm'].sigmoid_()
            wh = right_y['wh']
            reg = right_y['reg']

            heads = [hm, wh, reg]
            r_dets = self.ctdet_decode(heads, 40)
            # detections : ([batch_size, K, [xmin, ymin, xmax, ymax, score]])

            torch.cuda.synchronize()

        # TODO: create new img pair with detect results
        l_newinp = self.generate_newinp(x['l_img'], l_dets)
        r_newinp = self.generate_newinp(x['r_img'], r_dets)

        # TODO: feed each imgs through depth blocks
        imgl1, imgr1 = self.depth_model(l_inp, r_inp)
        carl1, carr1 = self.depth_model(l_newinp, r_newinp)

        # TODO: create cost volume
        left_cost, right_cost = self.depth_model.cost_volume(imgl1, imgr1, carl1, carr1)

        # TODO: 3d conv and deconv
        left_out = self.depth_model.after_cost(left_cost)
        right_out = self.depth_model.after_cost(right_cost)

        # TODO: softmax and return prob
        left_prob = F.softmax(-left_out, 1)
        right_prob = F.softmax(-right_out, 1)
        
        output = {'l_prob':left_prob, 'r_prob':right_prob,
                'l_newinp':l_newinp, 'r_newinp':r_newinp,
                'l_bbox':left_y, 'r_bbox':right_y}

        return output

    def find_top_k(self, heat, K=40):
        ''' Find top K key points (centers) in the headmap
        '''
        batch, cat, height, width = heat.size()
        topk_scores, topk_inds = torch.topk(heat.view(batch, cat, -1), K)
        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds // width).int().float()
        topk_xs = (topk_inds % width).int().float()
        topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
        topk_inds = self.gather_feat(
            topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
        topk_ys = self.gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
        topk_xs = self.gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

        return topk_score, topk_inds, topk_ys, topk_xs

    def ctdet_decode(self, heads, K=40):
        ''' Decoding the output

            Args:
                heads ([heatmap, width/height, regression]) - network results
            Return:
                detections([batch_size, K, [xmin, ymin, xmax, ymax, score]]) 
        '''
        heat, wh, reg = heads

        batch, cat, height, width = heat.size()

        scores, inds, ys, xs = self.find_top_k(heat, K)
        reg = self.transpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]

        wh = self.transpose_and_gather_feat(wh, inds)
        wh = wh.view(batch, K, 2)

        self.post_process(xs, ys, wh, reg)
        
        scores = scores.view(batch, K, 1)
        bboxes = torch.cat([xs - wh[..., 0:1] / 2, 
                            ys - wh[..., 1:2] / 2,
                            xs + wh[..., 0:1] / 2, 
                            ys + wh[..., 1:2] / 2], dim=2)
        detections = torch.cat([bboxes, scores], dim=2)
        
        return detections
    
    def gather_feat(self, feat, ind):

        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
            
        return feat
    
    def transpose_and_gather_feat(self, feat, ind):

        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self.gather_feat(feat, ind)

        return feat
    
    def post_process(self, xs, ys, wh, reg):
        ''' (Will modify args) Transfer all xs, ys, wh from heatmap size to input size
        '''
        for i in range(xs.size()[1]):
            xs[0, i, 0] = xs[0, i, 0] * 4
            ys[0, i, 0] = ys[0, i, 0] * 4
            wh[0, i, 0] = wh[0, i, 0] * 4
            wh[0, i, 1] = wh[0, i, 1] * 4

    def generate_newinp(self, img, boxes):
        
        # TODO: there are multiple bboxes
        # generate zero padding img
        # for each bbox, if index loc == car area,
        #   newinp[x][y] = inp img[x][y]

        # [xmin, ymin, xmax, ymax, score]

        boxes = boxes[:][:-1] # without score

        newinp = torch.zeros(3, Config.height, Config.width)

        for box in boxes:

            bbox = (int(torch.round(x)) for x in box)
            print(box)

            # plt.imshow('bbox', self.draw_bbox(img, box))
            # plt.xticks([]), plt.yticks([])
            # plt.show()

            # for x in range(Config.width):
            #     for y in range(Config.height):
                    # if (x >= box[0]) or (x <= box[2]) or (y <= box[3]) or (y >= box[1]):
                        # newinp[:][x][y] = img[:][x][y]

            newinp[:, bbox[0]:bbox[2], bbox[1]:bbox[3]] = img[:, bbox[0]:bbox[2], bbox[1]:bbox[3]]

        newinp = newinp.reshape(1, 3, Config.height, Config.width)

        return newinp

    # def draw_bbox(self, image, detections):
    #     ''' Given the original image and detections results (after threshold)
    #         Draw bounding boxes on the image
    #     '''
    #     image = np.asarray(image)
    #     height = image.shape[0]
    #     width = image.shape[1]
    #     for i in range(detections.shape[0]):
    #         cv2.rectangle(image, \
    #                       (detections[i, 0], detections[i, 1]), \
    #                       (detections[i, 2], detections[i, 3]), \
    #                       (0, 255, 0), 1)
    #
    #     return image
    
class BasicBlock(nn.Module):

    def __init__(self, in_planes, planes, stride=1):

        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()

    def forward(self, x):

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)

        return out

class ThreedConv(nn.Module):

    def __init__(self, in_planes, planes, stride=1):

        super(ThreedConv, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes,planes,kernel_size=3,stride=1,padding=1)
        self.bn3 = nn.BatchNorm3d(planes)

    def forward(self, x):

        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn3(self.conv3(out)))

        return out

class DepthBlock(nn.Module):

    def __init__(self, block, block_3d, num_block, feature_size=32):

        super(DepthBlock, self).__init__()

        self.maxdisp = int(Config.maxdisp/2)
        self.in_planes = 32
        self.feature_size = 32

        #first two conv2d
        self.conv0 = nn.Conv2d(3, 32, 5, 2, 2)
        self.bn0 = nn.BatchNorm2d(32)

        #res block
        self.res_block = self._make_layer(block, self.in_planes, 32, num_block[0], stride=1)

        #last conv2d
        self.conv1 = nn.Conv2d(32, 32, 3, 1, 1)

        #conv3d
        self.conv3d_1 = nn.Conv3d(64, 32, 3, 1, 1)
        self.bn3d_1 = nn.BatchNorm3d(32)
        self.conv3d_2 = nn.Conv3d(32, 32, 3, 1, 1)
        self.bn3d_2 = nn.BatchNorm3d(32)

        self.conv3d_3 = nn.Conv3d(64, 64, 3, 2, 1)
        self.bn3d_3 = nn.BatchNorm3d(64)
        self.conv3d_4 = nn.Conv3d(64, 64, 3, 2, 1)
        self.bn3d_4 = nn.BatchNorm3d(64)
        self.conv3d_5 = nn.Conv3d(64, 64, 3, 2, 1)
        self.bn3d_5 = nn.BatchNorm3d(64)

        #conv3d sub_sample block
        self.block_3d_1 = self._make_layer(block_3d, 64, 64, num_block[1],stride=2)
        self.block_3d_2 = self._make_layer(block_3d, 64, 64, num_block[1], stride=2)
        self.block_3d_3 = self._make_layer(block_3d, 64, 64, num_block[1], stride=2)
        self.block_3d_4 = self._make_layer(block_3d, 64, 128, num_block[1], stride=2)

        #deconv3d
        self.deconv1=nn.ConvTranspose3d(128, 64, 3, 2, 1, 1)
        self.debn1=nn.BatchNorm3d(64)
        self.deconv2 = nn.ConvTranspose3d(64, 64, 3, 2, 1, 1)
        self.debn2 = nn.BatchNorm3d(64)
        self.deconv3 = nn.ConvTranspose3d(64, 64, 3, 2, 1, 1)
        self.debn3 = nn.BatchNorm3d(64)
        self.deconv4 = nn.ConvTranspose3d(64, 32, 3, 2, 1, 1)
        self.debn4 = nn.BatchNorm3d(32)

        #last deconv3d
        self.deconv5 = nn.ConvTranspose3d(32, 1, 3, 2, 1, 1)
    
    def _make_layer(self, block, in_planes, planes, num_block, stride):
        strides = [stride] + [1] * (num_block - 1)
        layers = []
        for step in strides:
            layers.append(block(in_planes, planes, step))
        return nn.Sequential(*layers)

    def forward(self, imgLeft, imgRight):

        imgLeft = transforms.functional.resize(imgLeft, (Config.height // 2, Config.width // 2))
        imgRight = transforms.functional.resize(imgRight, (Config.height // 2, Config.width // 2))
        
        imgl0 = F.relu(self.bn0(self.conv0(imgLeft)))
        imgr0 = F.relu(self.bn0(self.conv0(imgRight)))

        imgl_block = self.res_block(imgl0)
        imgr_block = self.res_block(imgr0)

        imgl1 = self.conv1(imgl_block)
        imgr1 = self.conv1(imgr_block)

        return imgl1, imgr1

    def after_cost(self, cost_volume):

        conv3d_out = F.relu(self.bn3d_1(self.conv3d_1(cost_volume)))
        conv3d_out = F.relu(self.bn3d_2(self.conv3d_2(conv3d_out)))

        #conv3d block
        conv3d_block_1 = self.block_3d_1(cost_volume)
        conv3d_21 = F.relu(self.bn3d_3(self.conv3d_3(cost_volume)))
        conv3d_block_2 = self.block_3d_2(conv3d_21)
        conv3d_24 = F.relu(self.bn3d_4(self.conv3d_4(conv3d_21)))
        conv3d_block_3 = self.block_3d_3(conv3d_24)
        conv3d_27 = F.relu(self.bn3d_5(self.conv3d_5(conv3d_24)))
        conv3d_block_4 = self.block_3d_4(conv3d_27)
        
        #deconv
        deconv3d = F.relu(self.debn1(self.deconv1(conv3d_block_4))+conv3d_block_3)
        deconv3d = F.relu(self.debn2(self.deconv2(deconv3d))+conv3d_block_2)
        deconv3d = F.relu(self.debn3(self.deconv3(deconv3d))+conv3d_block_1)
        deconv3d = F.relu(self.debn4(self.deconv4(deconv3d))+conv3d_out)

        #last deconv3d
        deconv3d = self.deconv5(deconv3d)
        out = deconv3d.view(1, self.maxdisp, Config.height, Config.width)

        return out

    def cost_volume(self, imgl, imgr, carl, carr, feature_size=32):

        target_size = imgl.size() # 1, f, h//2, w//2 [1, 32, 128, 128]

        left_list = []
        right_list = []

        for d in range(self.maxdisp):

            if d == 0:

                leftplusright = torch.cat([imgl, carl, imgr, carr], axis=1)
                rightplusleft = torch.cat([imgr, carr, imgl, carl], axis=1)  # feature

            else:

                zero = torch.zeros(Config.batch_size, feature_size, Config.height // 4, d)

                rightmove = torch.cat([zero, imgr], axis=3)
                leftmove = torch.cat([imgl, zero], axis=3) # width

                rightmove1 = torch.cat([zero, carr], axis=3)
                leftmove1 = torch.cat([carl, zero], axis=3)

                # print(rightmove.size(), leftmove.size()) # [1, 32, 128, 128+d]

                rightmove = rightmove[:, :, :, :-d]
                leftmove = leftmove[:, :, :, d:]

                rightmove1 = rightmove1[:, :, :, :-d]
                leftmove1 = leftmove1[:, :, :, d:]

                # print(rightmove.size(), leftmove.size()) # [1, 32, 128, 128]

                leftplusright = torch.cat([imgl, rightmove, carl, rightmove1], axis=1)
                rightplusleft = torch.cat([imgr, leftmove, carr, leftmove1], axis=1)

                # print(leftplusright.size(), rightplusleft.size()) # [1, 128, 128, 128]

            left_list.append(leftplusright)
            right_list.append(rightplusleft) # len 80

        left_cost = torch.stack(left_list, axis=1)
        right_cost = torch.stack(right_list, axis=1)

        return left_cost, right_cost
