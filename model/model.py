import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import time

from dataset.transform import Transform
from dataset.load_dataset import Dataset
from model.detect import DlaNet
from config.config import Config


class Model(nn.Module):

    def __init__(self):

        print("Creating model...")

        super(Model, self).__init__()

        self.device = torch.device('cuda')

        self.transform = Transform()

        self.detect_model = torch.hub.load('ultralytics/yolov5', 'yolov5m')
        self.detect_model = self.detect_model.to(self.device)
        # higher mAP, half the params

        self.depth_model = DepthBlock(BasicBlock, ThreedConv, [8, 1])
        self.depth_model = self.depth_model.to(self.device)

    def forward(self, x):

        # TODO: detect vehicles in each image
        # (h, w, 3) -> (3, h, w) -> (1, 3, h, w)
        l_inp = x['l_img'].reshape(1, 3, Config.detect_height, Config.detect_width)
        r_inp = x['r_img'].reshape(1, 3, Config.detect_height, Config.detect_width)  # [1, 3, h, w]

        with torch.no_grad():
            # self.detect_model = DlaNet()
            # self.detect_model.load_state_dict(torch.load(Config.centernet_path))
            #
            # self.detect_model = self.detect_model.to(self.device)
            #
            # # freeze detection model's layers
            # for param in self.detect_model.parameters():
            #     param.requires_grad = False
            #
            # left_y = self.detect_model(l_inp)
            # right_y = self.detect_model(r_inp)
            #
            # del self.detect_model
            #
            # hm = left_y['hm'].sigmoid_()
            # wh = left_y['wh']
            # reg = left_y['reg']
            #
            # heads = [hm, wh, reg]
            # l_dets = self.ctdet_decode(heads, 40) # K is the number of remaining instances
            #
            # hm = right_y['hm'].sigmoid_()
            # wh = right_y['wh']
            # reg = right_y['reg']
            #
            # heads = [hm, wh, reg]
            # r_dets = self.ctdet_decode(heads, 40)
            # # detections : ([batch_size, K, [xmin, ymin, xmax, ymax, score]])
            #
            # del left_y, right_y
            #
            # l_dets.detach()
            # r_dets.detach()
            #
            # torch.cuda.synchronize()

            l_img = transforms.functional.to_pil_image(x['l_img'].view(3, Config.detect_height, Config.detect_width))
            r_img = transforms.functional.to_pil_image(x['r_img'].view(3, Config.detect_height, Config.detect_width))

            dets = self.detect_model([l_img, r_img])

            # TODO: create new img pair with detect results
            l_newinp = self.generate_newinp(x['l_img'], dets.xyxy[0])
            r_newinp = self.generate_newinp(x['r_img'], dets.xyxy[1])

        l_newinp = transforms.functional.resize(l_newinp, (Config.height, Config.width))
        r_newinp = transforms.functional.resize(r_newinp, (Config.height, Config.width))

        l_inp = transforms.functional.resize(x['l_img'], (Config.height, Config.width))
        r_inp = transforms.functional.resize(x['r_img'], (Config.height, Config.width))

        l_newinp = l_newinp.to(self.device)
        r_newinp = r_newinp.to(self.device)

        # inps size [ batch, 3, h, w ]

        # TODO: feed each imgs through depth blocks
        l_prob, r_prob = self.depth_model(l_inp, r_inp, l_newinp, r_newinp)

        output = {'l_pred': l_prob, 'r_pred': r_prob,
                  'l_newinp': l_newinp, 'r_newinp': r_newinp,
                  'l_bbox': dets.xyxy[0], 'r_bbox': dets.xyxy[1]}

        return output

    # def find_top_k(self, heat, K=40):
    #     ''' Find top K key points (centers) in the headmap
    #     '''
    #     batch, cat, height, width = heat.size()
    #     topk_scores, topk_inds = torch.topk(heat.view(batch, cat, -1), K)
    #     topk_inds = topk_inds % (height * width)
    #     topk_ys = (topk_inds // width).int().float()
    #     topk_xs = (topk_inds % width).int().float()
    #     topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    #     topk_inds = self.gather_feat(
    #         topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    #     topk_ys = self.gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    #     topk_xs = self.gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)
    #
    #     return topk_score, topk_inds, topk_ys, topk_xs
    #
    # def ctdet_decode(self, heads, K=40):
    #     ''' Decoding the output
    #
    #         Args:
    #             heads ([heatmap, width/height, regression]) - network results
    #         Return:
    #             detections([batch_size, K, [xmin, ymin, xmax, ymax, score]])
    #     '''
    #     heat, wh, reg = heads
    #
    #     batch, cat, height, width = heat.size()
    #
    #     scores, inds, ys, xs = self.find_top_k(heat, K)
    #     reg = self.transpose_and_gather_feat(reg, inds)
    #     reg = reg.view(batch, K, 2)
    #     xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
    #     ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    #
    #     wh = self.transpose_and_gather_feat(wh, inds)
    #     wh = wh.view(batch, K, 2)
    #
    #     self.post_process(xs, ys, wh, reg)
    #
    #     scores = scores.view(batch, K, 1)
    #     bboxes = torch.cat([xs - wh[..., 0:1] / 2,
    #                         ys - wh[..., 1:2] / 2,
    #                         xs + wh[..., 0:1] / 2,
    #                         ys + wh[..., 1:2] / 2], dim=2)
    #     detections = torch.cat([bboxes, scores], dim=2)
    #
    #     return detections

    # def gather_feat(self, feat, ind):
    #
    #     dim = feat.size(2)
    #     ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    #     feat = feat.gather(1, ind)
    #
    #     return feat
    #
    # def transpose_and_gather_feat(self, feat, ind):
    #
    #     feat = feat.permute(0, 2, 3, 1).contiguous()
    #     feat = feat.view(feat.size(0), -1, feat.size(3))
    #     feat = self.gather_feat(feat, ind)
    #
    #     return feat
    #
    # def post_process(self, xs, ys, wh, reg):
    #     ''' (Will modify args) Transfer all xs, ys, wh from heatmap size to input size
    #     '''
    #     for i in range(xs.size()[1]):
    #         xs[0, i, 0] = xs[0, i, 0] * 4
    #         ys[0, i, 0] = ys[0, i, 0] * 4
    #         wh[0, i, 0] = wh[0, i, 0] * 4
    #         wh[0, i, 1] = wh[0, i, 1] * 4

    def rescale_bbox(self, orig_shape, new_shape, bbox):

        x_scale = new_shape[1] / orig_shape[1]
        y_scale = new_shape[0] / orig_shape[0]

        xmin = int(torch.round(bbox[0] * x_scale))
        ymin = int(torch.round(bbox[1] * y_scale))
        xmax = int(torch.round(bbox[2] * x_scale))
        ymax = int(torch.round(bbox[3] * y_scale))

        return [xmin, ymin, xmax, ymax]

    def generate_newgt(self, gt, boxes):

        # gt [1, 375, 1242]

        # [xmin, ymin, xmax, ymax, confidence, class num, name]

        # gt = torch.squeeze(gt, 0)

        newimg = Image.new('L', Config.full_res_shape)
        gt_pil = transforms.functional.to_pil_image(gt)

        for box in boxes:

            if (box[5] == 2) or (box[5] == 5) or (box[5] == 7):
                # car, bus, truck

                # bbox = (int(torch.round(x)) for x in box)
                bbox = self.rescale_bbox((Config.detect_height, Config.detect_width),
                                         (Config.full_res_shape[1], Config.full_res_shape[0]), box[:4])
                bbox = list(bbox)

                if bbox[0] > bbox[2] or bbox[1] > bbox[3]:
                    continue

                to_paste = gt_pil.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
                # left, top, right, bottom (= xmin, ymin, xmax, ymax)

                newimg.paste(to_paste, box=(bbox[0], bbox[1]))

        newimg.save('{}{}.png'.format(Config.gen_newgt_path, time.time()))

        # newgt = transforms.ToTensor()(newimg)
        # newgt = newgt.view(1, 1, Config.full_res_shape[1], Config.full_res_shape[0])

        # newimg = newimg.resize((Config.full_res_shape[0], Config.full_res_shape[1]))
        newgt = transforms.ToTensor()(newimg)

        return newgt

    def generate_newpred(self, pred, boxes):

        # pred [1, 128, 256]

        newimg = Image.new('L', (Config.width, Config.height))
        pred_pil = transforms.functional.to_pil_image(pred)
        pred_pil.save('{}{}.png'.format(Config.gen_newpred_path_orig, time.time()))

        for box in boxes:

            if (box[5] == 2) or (box[5] == 5) or (box[5] == 7):
                # car, bus, truck

                # bbox = (int(torch.round(x)) for x in box)
                bbox = self.rescale_bbox((Config.detect_height, Config.detect_width), (Config.height, Config.width),
                                         box[:4])
                bbox = list(bbox)

                if bbox[0] > bbox[2] or bbox[1] > bbox[3]:
                    continue

                to_paste = pred_pil.crop((bbox[0], bbox[1], bbox[2], bbox[3]))  # left, top, right, bottom

                newimg.paste(to_paste, box=(bbox[0], bbox[1]))

        newimg.save('{}{}.png'.format(Config.gen_newpred_path_final, time.time()))

        newpred = transforms.ToTensor()(newimg)
        newpred = newpred.view(Config.batch_size, Config.height, Config.width)  # [1, 128, 128]

        return newpred

    def generate_newinp(self, img, boxes):

        # [xmin, ymin, xmax, ymax, score]

        img = torch.squeeze(img)

        newimg = Image.new('RGB', (Config.width, Config.height))
        img_pil = transforms.functional.to_pil_image(img)

        for box in boxes:

            if (box[5] == 2) or (box[5] == 5) or (box[5] == 7):

                bbox = self.rescale_bbox((Config.detect_height, Config.detect_width), (Config.height, Config.width),
                                         box[:4])
                bbox = list(bbox)

                if bbox[0] > bbox[2] or bbox[1] > bbox[3]:
                    continue

                one_car = img_pil.crop((bbox[0], bbox[1], bbox[2], bbox[3]))  # left, top, right, bottom

                newimg.paste(one_car, box=(bbox[0], bbox[1]))

        newimg.save('{}{}.png'.format(Config.gen_newinp_path, time.time()))

        newinp = transforms.ToTensor()(newimg)
        newinp = newinp.view(Config.batch_size, 3, Config.height, Config.width)

        return newinp


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


#
class ThreedConv(nn.Module):

    def __init__(self, in_planes, planes, stride=1):
        super(ThreedConv, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm3d(planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn3(self.conv3(out)))

        return out


# modified
class DepthBlock(nn.Module):

    def __init__(self, block, block_3d, num_block, feature_size=32):

        super(DepthBlock, self).__init__()

        self.maxdisp = int(Config.maxdisp / 2)
        self.in_planes = 32
        self.feature_size = feature_size

        # first conv
        self.conv0 = nn.Conv2d(3, 32, 5, 2, 2)
        self.bn0 = nn.BatchNorm2d(32)

        # res block
        self.res_block = self._make_layer(block, self.in_planes, 32, num_block[0], stride=1)

        # last conv2d
        self.conv1 = nn.Conv2d(32, 32, 3, 1, 1)

        # conv3d
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

        # conv3d sub_sample block
        self.block_3d_1 = self._make_layer(block_3d, 64, 64, num_block[1], stride=2)
        self.block_3d_2 = self._make_layer(block_3d, 64, 64, num_block[1], stride=2)
        self.block_3d_3 = self._make_layer(block_3d, 64, 64, num_block[1], stride=2)
        self.block_3d_4 = self._make_layer(block_3d, 64, 128, num_block[1], stride=2)

        # deconv3d
        self.deconv1 = nn.ConvTranspose3d(128, 64, 3, 2, 1, 1)
        self.debn1 = nn.BatchNorm3d(64)
        self.deconv2 = nn.ConvTranspose3d(64, 64, 3, 2, 1, 1)
        self.debn2 = nn.BatchNorm3d(64)
        self.deconv3 = nn.ConvTranspose3d(64, 64, 3, 2, 1, 1)
        self.debn3 = nn.BatchNorm3d(64)
        self.deconv4 = nn.ConvTranspose3d(64, 32, 3, 2, 1, 1)
        self.debn4 = nn.BatchNorm3d(32)

        # last deconv3d
        self.deconv5 = nn.ConvTranspose3d(32, 1, 3, 2, 1, 1)

    def _make_layer(self, block, in_planes, planes, num_block, stride):
        strides = [stride] + [1] * (num_block - 1)
        layers = []
        for step in strides:
            layers.append(block(in_planes, planes, step))
        return nn.Sequential(*layers)

    def precost(self, x):

        x = F.relu(self.bn0(self.conv0(x)))
        x = self.res_block(x)
        x = self.conv1(x)

        return x

    def forward(self, imgl, imgr, carl, carr):

        imgl0 = self.precost(imgl)
        imgr0 = self.precost(imgr)
        carl0 = self.precost(carl)
        carr0 = self.precost(carr)

        # TODO: create cost volume
        left_cost, right_cost = self.cost_volume(imgl0, imgr0, carl0, carr0)

        del imgl0, imgr0, carl0, carr0

        # TODO: 3d conv and deconv
        left_out = self.after_cost(left_cost)
        right_out = self.after_cost(right_cost)

        # # TODO: squeeze, apply softmax, and return prob
        left_squeeze = torch.squeeze(left_out)
        right_squeeze = torch.squeeze(right_out)

        left_softmax = F.softmax(-left_squeeze, dim=0)
        right_softmax = F.softmax(-right_squeeze, dim=0)

        d_grid = torch.arange(Config.maxdisp * 2, dtype=torch.float32).reshape(-1, 1, 1).cuda()
        d_grid = d_grid.repeat(1, Config.height, Config.width)

        # TODO: soft argmin
        left_prob = torch.sum(torch.mul(left_softmax, d_grid), dim=0, keepdim=True)
        right_prob = torch.sum(torch.mul(right_softmax, d_grid), dim=0, keepdim=True)

        return left_prob, right_prob

    def after_cost(self, cost_volume):

        conv3d_out = F.relu(self.bn3d_1(self.conv3d_1(cost_volume)))
        conv3d_out = F.relu(self.bn3d_2(self.conv3d_2(conv3d_out)))

        # conv3d block
        conv3d_block_1 = self.block_3d_1(cost_volume)
        conv3d_21 = F.relu(self.bn3d_3(self.conv3d_3(cost_volume)))
        conv3d_block_2 = self.block_3d_2(conv3d_21)
        conv3d_24 = F.relu(self.bn3d_4(self.conv3d_4(conv3d_21)))
        conv3d_block_3 = self.block_3d_3(conv3d_24)
        conv3d_27 = F.relu(self.bn3d_5(self.conv3d_5(conv3d_24)))
        conv3d_block_4 = self.block_3d_4(conv3d_27)

        # deconv
        deconv3d = F.relu(self.debn1(self.deconv1(conv3d_block_4)) + conv3d_block_3)
        deconv3d = F.relu(self.debn2(self.deconv2(deconv3d)) + conv3d_block_2)
        deconv3d = F.relu(self.debn3(self.deconv3(deconv3d)) + conv3d_block_1)
        deconv3d = F.relu(self.debn4(self.deconv4(deconv3d)) + conv3d_out)

        # last deconv3d
        deconv3d = self.deconv5(deconv3d)  # [1, 1, 256, h, w] (1, 1, 256, 128, 256)
        out = deconv3d.view(1, Config.maxdisp * 2, Config.height, Config.width)
        prob = F.softmax(-out, 1)

        return prob

    def cost_volume(self, imgl, imgr, carl, carr, feature_size=32):

        # img: 1, f, h//2, w//2

        left_list = []
        right_list = []

        for d in range(self.maxdisp):

            if d == 0:

                leftplusright = torch.cat([imgl, carl, imgr, carr], axis=1)
                rightplusleft = torch.cat([imgr, carr, imgl, carl], axis=1)  # feature

                # _ plus _ size: [1, f, h//2, w//2] (1, 32, 64, 128)

            else:

                zero = torch.zeros(Config.batch_size, feature_size, Config.height // 2, d).cuda()  # [1, 32, 64, d]
                # [1, f, h//2, d] (1, 32, 64, d)

                rightmove = torch.cat([zero, imgr], axis=3)
                leftmove = torch.cat([imgl, zero], axis=3)
                # [1, f, h//2, maxdisp + d]

                rightmove1 = torch.cat([zero, carr], axis=3)
                leftmove1 = torch.cat([carl, zero], axis=3)
                # [1, f, h//2, maxdisp + d]

                rightmove = rightmove[:, :, :, :-d]
                leftmove = leftmove[:, :, :, d:]
                # [1, f, h//2, maxdisp]

                rightmove1 = rightmove1[:, :, :, :-d]
                leftmove1 = leftmove1[:, :, :, d:]
                # [1, f, h//2, maxdisp] (1, 32, 64, 128)

                leftplusright = torch.cat([imgl, rightmove, carl, rightmove1], axis=1)
                rightplusleft = torch.cat([imgr, leftmove, carr, leftmove1], axis=1)
                # (1, 128, 64, 128)

            left_list.append(leftplusright)
            right_list.append(rightplusleft)

        left_cost = torch.stack(left_list, axis=1)
        right_cost = torch.stack(right_list, axis=1)
        # (1, 64, 128, 64, 182)

        return left_cost, right_cost
