import os
import numpy as np
from kitti_utils import generate_depth_map
import skimage.transform

class Dataset(object):

    def __init__(self, kitti_dir, split):

        self.datasetDir = kitti_dir
        self.img_train = []
        self.img_val = []
        self.split = split
        self.depthGt_train = []
        self.depthGt_val = []

    def getImgPath(self):

        f_train = open(f'splits/{self.split}/train_files.txt', 'r')
        lines = f_train.readlines()
        for line in lines:
            line = line.split(' ')
            if line[2] == 'l':
                line = os.path.join(line[0], '/image_02/data/{:010d}.jpg'.format(line[1]))
                self.img_train.append(f'kitti_data/images/{line}')
        f_train.close()

        print(f'Loaded path of {self.img_train.len()} train images')

        f_val = open(f'splits/{self.split}/val_files.txt', 'r')
        lines = f_val.readlines()
        for line in lines:
            line = line.split(' ')
            if line[2] == 'l':
                line = os.path.join(line[0], '/image_02/data/{:010d}.jpg'.format(line[1]))
                self.img_val.append(f'kitti_data/images/{line}')
        f_val.close()

        print(f'Loaded path of {self.img_val.len()} validation images')

    def getDepthGtPath(self):

        f_train = open(f'splits/{self.split}/train_files.txt', 'r')
        lines = f_train.readlines()
        for line in lines:
            line = line.split(' ')
            if line[2] == 'l':
                line = os.path.join(line[0], '/velodyne_points/data/{:010d}.bin'.format(line[1]))
                self.depthGt_train.append(f'kitti_data/images/{line}')
        f_train.close()

        print(f'Loaded path of {self.depthGt_train.len()} train depth gt')

        f_val = open(f'splits/{self.split}/val_files.txt', 'r')
        lines = f_val.readlines()
        for line in lines:
            line = line.split(' ')
            if line[2] == 'l':
                line = os.path.join(line[0], '/velodyne_points/data/{:010d}.bin'.format(line[1]))
                self.depthGt_val.append(f'kitti_data/images/{line}')
        f_val.close()

        print(f'Loaded path of {self.depthGt_val.len()} validation depth gt')

    def getGtDepthMap(self):

        for path in self.depthGt_train:
            calib_path = os.path.join(self.data_path, folder.split("/")[0])

            depth_gt = generate_depth_map(calib_path, velo_filename, self.side_map[side])
            depth_gt = skimage.transform.resize(depth_gt, self.full_res_shape[::-1], order=0, preserve_range=True, mode='constant')

    def getDetectionGtPath(self):

