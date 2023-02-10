from .backbone.renset import ResNet
from model.decoder import Decoder
from model.head import Head
from model.fpn import FPN
from loss.utils import map2coords
import torch
from torch import nn
import torch.nn.functional as F

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

class ThreeDConv(nn.Module):

    def __init__(self, in_planes, planes, stride=1):
        super(ThreeDConv, self).__init__()
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

class Depth(nn.Module):

    def __init__(self, cfg, block, block_3d, num_block, h, w, maxdisp):

        super(Depth, self).__init__()

        self._fpn = cfg.fpn
        self.down_stride = cfg.down_stride
        self.score_th = cfg.score_th
        self.CLASSES_NAME = cfg.CLASSES_NAME

        self.height = h
        self.width = w
        self.maxdisp = int(maxdisp/2)
        self.in_places = 32
        
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

    def forward(self, left_img, right_img):

        img_l_0 = F.relu(self.bn0(self.conv0(left_img)))
        img_r_0 = F.relu(self.bn0(self.conv0(right_img)))

        left_imgblock = self.res_block(img_l_0)
        right_imgblock = self.res_block(img_r_0)

        img_l_1 = self.conv1(left_imgblock)
        img_r_1 = self.conv1(right_imgblock)

        # cost volume
        cost = self.cost_volume(img_l_1)
        conv3d_out = F.relu(self.bn3d_1(self.conv3d_1(cost)))
        conv3d_out = F.relu(self.bn3d_2(self.conv3D_2(conv3d_out)))

        prob = F.softmax(-conv3d_out, 1)

        return prob

    def _make_layer(self,block,in_planes,planes,num_block,stride):
        strides=[stride]+[1]*(num_block-1)
        layers=[]
        for step in strides:
            layers.append(block(in_planes,planes,step))
        return nn.Sequential(*layers)

    def cost_volume(self, imgl, imgr, lr='left'):

        xx_list = []

        if lr == 'left':
            pad_opr1 = nn.ZeroPad2d((0, self.maxdisp, 0, 0))
            xleft = pad_opr1(imgl)

            for d in range(self.maxdisp):
                pad_opr2 = nn.ZeroPad2d((d, self.maxdisp - d, 0, 0))
                xright = pad_opr2(imgr)
                xx_temp = torch.cat((xleft, xright), 1)
                xx_list.append(xx_temp)

        elif lr == 'right':
            pad_opr1 = nn.ZeroPad2d((0, self.maxdisp, 0, 0))
            xright = pad_opr1(imgr)

            for d in range(self.maxdisp):
                pad_opr2 = nn.ZeroPad2d((d, self.maxdisp - d, 0, 0))
                xleft = pad_opr2(imgl)
                xx_temp = torch.cat((xleft, xright), 1)
                xx_list.append(xx_temp)

        xx = torch.cat(xx_list, 1)
        xx = xx.view(1, self.maxdisp, 64, int(self.height / 2), int(self.width / 2) + self.maxdisp)
        xx0 = xx.permute(0,2,1,3,4)
        xx0 = xx0[:, :, :, :, :int(self.width / 2)]
        
        return xx0