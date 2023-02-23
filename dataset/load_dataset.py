import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms

from .transform import Transform
from config.config import Config

class Dataset(data.Dataset):

    def __init__(self, is_train=True):

        super(Dataset, self).__init__()

        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        fullres_shape = (1242, 375)

        self.is_train = True

        self.dataset_path = Config.dataset_path
        self.height = Config.height
        self.width = Config.width

        self.images = {'l':[], 'r':[]}
        self.depths = {'l':[], 'r':[]}
        
        self.transform = Transform

        self.get_data_from_dir()

    def __len__(self):

        return len(self.images['l'])
        
    def __getitem__(self, idx):

        l_img = Image.open(self.images['l'][idx])
        r_img = Image.open(self.images['r'][idx])

        l_img, r_img = self.transform(l_img), self.transform(r_img)

        # no resize for depth
        l_depth = Image.open(self.depths['l'][idx])
        r_depth = Image.open(self.depths['r'][idx])
        l_depth = torch.from_numpy((l_depth / 255.).astype(np.float32))
        r_depth = torch.from_numpy((r_depth / 255.).astype(np.float32))

        item = {'l_img': l_img, 'r_img': r_img,
                'l_depth': l_depth, 'r_depth': r_depth}
            
        return item
        
    def get_data_from_dir(self):

            depth_folder = '../dataset/data_depth_annotated/{}'.format('train' if self.is_train else 'val')

            for sync in os.listdir(depth_folder):

                date = sync[:10]
                         
                depth_img_folder = os.path.join(sync, 'proj_depth/ground_truth/image_0{}')

                for img_num in os.listdir(os.path.join(depth_folder, depth_folder(2))):
                            
                    full_img_path = os.path.join(self.dataset_path, date, sync, 'image_0{}/data', '{:010d}.png'.format(img_num))
                    full_depth_path = os.path.join(depth_folder, depth_img_folder, '{:010d}.png'.format(img_num))

                    self.images['l'].append(full_img_path(2))
                    self.images['r'].append(full_img_path(3))

                    l_depth = Image.open(full_depth_path(2))
                    r_depth = Image.open(full_depth_path(3))

                    self.depths['l'].append(l_depth)
                    self.depths['r'].append(r_depth)
                        
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