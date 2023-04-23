import os, cv2, scipy
import numpy as np
from PIL import Image
import torch.utils.data as data
from collections import Counter
import skimage.transform
import time
from scipy.interpolate import LinearNDInterpolator
import torch
from skimage.color import rgb2gray

from .transform import Transform
from config.config import Config

class Dataset(data.Dataset):

    def __init__(self):

        super(Dataset, self).__init__()

        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.dataset_path = Config.dataset_path
        self.height = Config.height
        self.width = Config.width

        self.images = {'l': [], 'r': []}
        
        self.transform = Transform()

        self.get_data_from_dir()

        self.full_res_shape = (1242, 375)

        self.idx = None

    def __len__(self):

        return len(self.images['l'])
        
    def __getitem__(self, idx):

        self.idx = idx

        split_path1 = (self.images['l'][idx]).split('/')
        split_path = split_path1[2].split('\\')

        date = split_path[1]
        sync = split_path[2]
        n = split_path1[-1].split('.')[0].split('\\')[1]
        num = split_path1[-1].split('.')[0] + '.bin'

        # print(n)

        # TODO: load velo points
        calib_path = os.path.join(self.dataset_path, date)
        velo_filename = os.path.join(calib_path, sync, 'velodyne_points', num)

        if not os.path.exists(velo_filename):

            idx = 0
            velo_filename = os.path.join(calib_path, sync, 'velodyne_points', 'data', '0000000000.bin')

        l_gt = self.generate_depth_map(calib_path, velo_filename, 2)
        r_gt = self.generate_depth_map(calib_path, velo_filename, 3)
        l_gt = skimage.transform.resize(l_gt, self.full_res_shape[::-1], order=0, preserve_range=True, mode='constant')
        r_gt = skimage.transform.resize(r_gt, self.full_res_shape[::-1], order=0, preserve_range=True, mode='constant')

        (Image.fromarray(l_gt).convert('L')).save('./dense-depth/{}.png'.format(time.time()))

        l_gt = self.normalize_gt(l_gt)
        r_gt = self.normalize_gt(r_gt)

        l_gt = torch.from_numpy(l_gt)
        r_gt = torch.from_numpy(r_gt)

        l_img = Image.open(self.images['l'][idx]).convert('RGB')
        r_img = Image.open(self.images['r'][idx]).convert('RGB')

        l_img = self.transform(l_img)
        r_img = self.transform(r_img)

        item = {'l_img': l_img, 'r_img': r_img, 'l_depth': l_gt, 'r_depth': r_gt}

        return item

    def normalize_gt(self, gt):

        min_depth = np.min(gt)
        max_depth = np.max(gt)

        gt_norm = (gt - min_depth) / (max_depth - min_depth)

        return gt_norm

    def generate_depth_map(self, calib_path, velo_filename, interp=True, cam=2, vel_depth=False):

        # load calibration files
        cam2cam = self.read_calib_file(os.path.join(calib_path, 'calib_cam_to_cam.txt'))
        velo2cam = self.read_calib_file(os.path.join(calib_path, 'calib_velo_to_cam.txt'))
        velo2cam = np.hstack((velo2cam['R'].reshape(3, 3), velo2cam['T'][..., np.newaxis]))
        velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))

        # get image shape
        im_shape = cam2cam["S_rect_02"][::-1].astype(np.int32)

        # compute projection matrix velodyne->image plane
        R_cam2rect = np.eye(4)
        R_cam2rect[:3, :3] = cam2cam['R_rect_00'].reshape(3, 3)
        P_rect = cam2cam['P_rect_0' + str(cam)].reshape(3, 4)
        P_velo2im = np.dot(np.dot(P_rect, R_cam2rect), velo2cam)

        # load velodyne points and remove all behind image plane (approximation)
        # each row of the velodyne data is forward, left, up, reflectance
        velo = self.load_velodyne_points(velo_filename)
        velo = velo[velo[:, 0] >= 0, :]

        # project the points to the camera
        velo_pts_im = np.dot(P_velo2im, velo.T).T
        velo_pts_im[:, :2] = velo_pts_im[:, :2] / velo_pts_im[:, 2][..., np.newaxis]

        if vel_depth:
            velo_pts_im[:, 2] = velo[:, 0]

        # check if in bounds
        # use minus 1 to get the exact same value as KITTI matlab code
        velo_pts_im[:, 0] = np.round(velo_pts_im[:, 0]) - 1
        velo_pts_im[:, 1] = np.round(velo_pts_im[:, 1]) - 1
        val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
        val_inds = val_inds & (velo_pts_im[:, 0] < im_shape[1]) & (velo_pts_im[:, 1] < im_shape[0])
        velo_pts_im = velo_pts_im[val_inds, :]

        # project to image
        depth = np.zeros((im_shape[:2]))
        depth[velo_pts_im[:, 1].astype(np.int), velo_pts_im[:, 0].astype(np.int)] = velo_pts_im[:, 2]

        # find the duplicate points and choose the closest depth
        inds = self.sub2ind(depth.shape, velo_pts_im[:, 1], velo_pts_im[:, 0])
        dupe_inds = [item for item, count in Counter(inds).items() if count > 1]
        for dd in dupe_inds:
            pts = np.where(inds == dd)[0]
            x_loc = int(velo_pts_im[pts[0], 0])
            y_loc = int(velo_pts_im[pts[0], 1])
            depth[y_loc, x_loc] = velo_pts_im[pts, 2].min()
        depth[depth < 0] = 0

        # interpolate the depth map to fill in holes
        depth = self.lin_interp(im_shape, velo_pts_im)

        return depth

    def lin_interp(self, shape, xyd):

        # taken from https://github.com/hunse/kitti
        m, n = shape
        ij, d = xyd[:, 1::-1], xyd[:, 2]
        f = LinearNDInterpolator(ij, d, fill_value=0)
        J, I = np.meshgrid(np.arange(n), np.arange(m))
        IJ = np.vstack([I.flatten(), J.flatten()]).T
        disparity = f(IJ).reshape(shape)

        return disparity

    def read_calib_file(self, path):
        """Read KITTI calibration file
        (from https://github.com/hunse/kitti)
        """
        float_chars = set("0123456789.e+- ")
        data = {}
        with open(path, 'r') as f:
            for line in f.readlines():
                key, value = line.split(':', 1)
                value = value.strip()
                data[key] = value
                if float_chars.issuperset(value):
                    # try to cast to float array
                    try:
                        data[key] = np.array(list(map(float, value.split(' '))))
                    except ValueError:
                        # casting error: data[key] already eq. value, so pass
                        pass

        return data

    def load_velodyne_points(self, filename):
        """Load 3D point cloud from KITTI file format
        (adapted from https://github.com/hunse/kitti)
        """
        points = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
        points[:, 3] = 1.0  # homogeneous
        return points

    def sub2ind(self, matrixSize, rowSub, colSub):
        """Convert row, col matrix subscripts to linear indices
        """
        m, n = matrixSize
        return rowSub * (n - 1) + colSub - 1

    def get_data_from_dir(self):

        date_list = os.listdir(self.dataset_path) # ../dataset/raw_data/date
        date_list = date_list[:-1]

        for date in date_list:

            sync_list = os.listdir(os.path.join(self.dataset_path, date))
            sync_list = sync_list[:-3]

            for sync in sync_list:

                data_path = os.path.join(self.dataset_path, date, sync, 'image_02/data')

                imgs = os.listdir(data_path)
                for img in imgs:

                    img_path = os.path.join(data_path, img)
                    self.images['l'].append(img_path)

                data_path = os.path.join(self.dataset_path, date, sync, 'image_03/data')

                imgs = os.listdir(data_path.format(3))
                for img in imgs:

                    img_path = os.path.join(data_path, img)
                    self.images['r'].append(img_path)
'''
#Values    Name      Description
----------------------------------------------------------------------------
   1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                     'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                     'Misc' or 'DontCare'
   1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                     truncated refers to the object leaving image boundaries
   1    occluded     Integer (0,1,2,3) indicating occlusion state:
                     0 = fully visible, 1 = partly occluded
                     2 = largely occluded, 3 = unknown
   1    alpha        Observation angle of object, ranging [-pi..pi]
   4    bbox         2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates
   3    dimensions   3D object dimensions: height, width, length (in meters)
   3    location     3D object location x,y,z in camera coordinates (in meters)
   1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
   1    score        Only for results: Float, indicating confidence in
                     detection, needed for p/r curves, higher is better.
'''