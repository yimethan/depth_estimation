import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.functional as F

from model.detect import DlaNet
from config.config import Config
from tensorboardX import SummaryWriter

class Model(nn.Module):

    def __init__(self):

        print("Creating model...")

        self.detect_model = DlaNet()
        self.detect_model.load_state_dict(Config.centernet_path)

        # freeze detection model's layers
        for param in self.detect_model.parameters():
            param.requires_grad = False

        # gc based
        self.depth_model = DepthBlock(BasicBlock, ThreedConv, [8, 1],
                                    Config.height, Config.width, Config.maxdisp)

    def forward(self, x):
        
        # TODO: detect vehicles in each image
        # (h, w, 3) -> (3, h, w) -> (1, 3, h, w)
        l_inp = x['l_img'].transpose(1, 2, 0).reshape(1, 3, Config.height, Config.width)
        r_inp = x['r_img'].transpose(1, 2, 0).reshape(1, 3, Config.height, Config.width)

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
        l_newinp = self.generate_newinp(l_inp, l_dets[2])
        r_newinp = self.generate_newinp(r_inp, r_dets[2])

        # TODO: feed each imgs through depth blocks
        imgl1, imgr1 = self.depth_model(l_inp, r_inp)
        carl1, carr1 = self.depth_model(l_newinp, r_newinp)

        # TODO: create cost volume
        left_cost, right_cost = DepthBlock.cost_volume(imgl1, imgr1, carl1, carr1)

        # TODO: 3d conv and deconv
        left_out = DepthBlock.after_cost(left_cost)
        right_out = DepthBlock.after_cost(right_cost)

        # TODO: softmax and return prob
        left_prob = F.softmax(-left_out, 1)
        right_prob = F.softmax(-right_out, 1)
        
        output = {'l_prob':left_prob, 'r_prob':right_prob,
                'l_newinp':l_newinp, 'r_newinp':r_newinp,
                'l_bbox':left_y, 'r_bbox':right_y}

        return output

    def ctdet_decode(self, heads, K = 40):
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
    
    def gather_feat(feat, ind):

        dim  = feat.size(2)
        ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
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

    def generate_newinp(inp, box):
        
        # TODO: there are multiple bboxes
        # generate zero padding img
        # for each bbox, if index loc == car area,
        #   newinp[x][y] = inp img[x][y]

        size = torch.Size(inp)
        newinp = torch.zeros(size[0], size[1])

        for x in Config.width:

            for y in Config.height:

                if (x >= box[0]) or (x <= box[2]) or (y <= box[3]) or (y >= box[1]):
                    
                    newinp[x][y] = inp[x][y]

        return newinp
    
class BasicBlock(nn.Module):

    def __init__(self,in_planes,planes,stride=1):

        super(BasicBlock,self).__init__()
        self.conv1=nn.Conv2d(in_planes,planes,kernel_size=3,stride=stride,padding=1)
        self.bn1=nn.BatchNorm2d(planes)
        self.conv2=nn.Conv2d(planes,planes,kernel_size=3,stride=1,padding=1)
        self.bn2=nn.BatchNorm2d(planes)
        self.shortcut=nn.Sequential()

    def forward(self, x):

        out=F.relu(self.bn1(self.conv1(x)))
        out=self.bn2(self.conv2(out))
        out+=self.shortcut(x)
        out=F.relu(out)

        return out

class ThreedConv(nn.Module):

    def __init__(self,in_planes,planes,stride=1):

        super(ThreedConv, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3=nn.Conv3d(planes,planes,kernel_size=3,stride=1,padding=1)
        self.bn3=nn.BatchNorm3d(planes)

    def forward(self, x):

        out=F.relu(self.bn1(self.conv1(x)))
        out=F.relu(self.bn2(self.conv2(out)))
        out=F.relu(self.bn3(self.conv3(out)))

        return out

class DepthBlock(nn.Module):

    def __init__(self, block, block_3d, num_block, h, w, maxdisp):

        super(Model, self).__init__()

        self.height = h
        self.width = w
        self.maxdisp = int(maxdisp/2)
        self.in_planes = 32

        #first two conv2d
        self.conv0=nn.Conv2d(3,32,5,2,2)
        self.bn0=nn.BatchNorm2d(32)

        #res block
        self.res_block=self._make_layer(block,self.in_planes,32,num_block[0],stride=1)

        #last conv2d
        self.conv1=nn.Conv2d(32,32,3,1,1)

        #conv3d
        self.conv3d_1=nn.Conv3d(64,32,3,1,1)
        self.bn3d_1=nn.BatchNorm3d(32)
        self.conv3d_2=nn.Conv3d(32,32,3,1,1)
        self.bn3d_2=nn.BatchNorm3d(32)

        self.conv3d_3=nn.Conv3d(64,64,3,2,1)
        self.bn3d_3=nn.BatchNorm3d(64)
        self.conv3d_4=nn.Conv3d(64,64,3,2,1)
        self.bn3d_4=nn.BatchNorm3d(64)
        self.conv3d_5=nn.Conv3d(64,64,3,2,1)
        self.bn3d_5=nn.BatchNorm3d(64)

        #conv3d sub_sample block
        self.block_3d_1 = self._make_layer(block_3d, 64, 64, num_block[1],stride=2)
        self.block_3d_2 = self._make_layer(block_3d, 64, 64, num_block[1], stride=2)
        self.block_3d_3 = self._make_layer(block_3d, 64, 64, num_block[1], stride=2)
        self.block_3d_4 = self._make_layer(block_3d, 64, 128, num_block[1], stride=2)

        #deconv3d
        self.deconv1=nn.ConvTranspose3d(128,64,3,2,1,1)
        self.debn1=nn.BatchNorm3d(64)
        self.deconv2 = nn.ConvTranspose3d(64, 64, 3, 2, 1, 1)
        self.debn2 = nn.BatchNorm3d(64)
        self.deconv3 = nn.ConvTranspose3d(64, 64, 3, 2, 1, 1)
        self.debn3 = nn.BatchNorm3d(64)
        self.deconv4 = nn.ConvTranspose3d(64, 32, 3, 2, 1, 1)
        self.debn4 = nn.BatchNorm3d(32)

        #last deconv3d
        self.deconv5 = nn.ConvTranspose3d(32, 1, 3, 2, 1, 1)
    
    def _make_layer(self,block,in_planes,planes,num_block,stride):
        strides=[stride]+[1]*(num_block-1)
        layers=[]
        for step in strides:
            layers.append(block(in_planes,planes,step))
        return nn.Sequential(*layers)

    def forward(self, imgLeft, imgRight):
        
        imgl0=F.relu(self.bn0(self.conv0(imgLeft)))
        imgr0=F.relu(self.bn0(self.conv0(imgRight)))

        imgl_block=self.res_block(imgl0)
        imgr_block=self.res_block(imgr0)

        imgl1=self.conv1(imgl_block)
        imgr1=self.conv1(imgr_block)

        return imgl1, imgr1

    def after_cost(self, cost_volume):

        conv3d_out=F.relu(self.bn3d_1(self.conv3d_1(cost_volume)))
        conv3d_out=F.relu(self.bn3d_2(self.conv3d_2(conv3d_out)))

        #conv3d block
        conv3d_block_1=self.block_3d_1(cost_volume)
        conv3d_21=F.relu(self.bn3d_3(self.conv3d_3(cost_volume)))
        conv3d_block_2=self.block_3d_2(conv3d_21)
        conv3d_24=F.relu(self.bn3d_4(self.conv3d_4(conv3d_21)))
        conv3d_block_3=self.block_3d_3(conv3d_24)
        conv3d_27=F.relu(self.bn3d_5(self.conv3d_5(conv3d_24)))
        conv3d_block_4=self.block_3d_4(conv3d_27)
        
        #deconv
        deconv3d=F.relu(self.debn1(self.deconv1(conv3d_block_4))+conv3d_block_3)
        deconv3d=F.relu(self.debn2(self.deconv2(deconv3d))+conv3d_block_2)
        deconv3d=F.relu(self.debn3(self.deconv3(deconv3d))+conv3d_block_1)
        deconv3d=F.relu(self.debn4(self.deconv4(deconv3d))+conv3d_out)

        #last deconv3d
        deconv3d=self.deconv5(deconv3d)
        out=deconv3d.view(1, self.maxdisp*2, self.height, self.width)

        return out

    def cost_volume(self, imgl, imgr, carl, carr):

        xx_list = []

        pad_opr1 = nn.ZeroPad2d((0, self.maxdisp, 0, 0))
        pad_opr2 = nn.ZeroPad2d((d, self.maxdisp - d, 0, 0))
        
        # left + right
        xleft = pad_opr1(imgl)
        yleft = pad_opr1(carl)

        for d in range(self.maxdisp):
            xright = pad_opr2(imgr)
            yright = pad_opr2(carr)
            xx_temp = torch.cat((xleft, yleft, xright, yright), 1)
            xx_list.append(xx_temp)

        xx = torch.cat(xx_list, 1)
        xx = xx.view(1, self.maxdisp, 64, int(self.height / 2), int(self.width / 2) + self.maxdisp)
        xx_left = xx.permute(0,2,1,3,4)
        xx_left = xx_left[:, :, :, :, :int(self.width / 2)]

        xx_list = []

        pad_opr3 = nn.ZeroPad2d((self.maxdisp, 0, 0, 0))
        pad_opr4 = nn.ZeroPad2d((self.maxdisp - d, d, 0, 0))

        # right + left
        xright = pad_opr3(imgr)
        yright = pad_opr3(carr)

        for d in range(self.maxdisp):
            xleft = pad_opr4(imgl)
            yleft = pad_opr4(carl)
            xx_temp = torch.cat((xright, yright, xleft, yleft), 1)
            xx_list.apend(xx_temp)

        xx = torch.cat(xx_list, 1)
        xx = xx.view(1, self.maxdisp, 64, int(self.height / 2), int(self.width / 2) + self.maxdisp)
        xx_right = xx.permute(0,2,1,3,4)
        xx_right = xx_right[:, :, :, :, :int(self.width / 2)]

        return xx_left, xx_right