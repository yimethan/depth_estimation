import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms

from config.config import Config

class Dataset(data.Dataset):

    def __init__(self, filenames, is_train=True):

        super(Dataset, self).__init__()

        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        # self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}
        fullres_shape = (1242, 375)

        self.is_train = True
        self.filenames = filenames

        self.dataset_path = Config.dataset_path
        self.height = Config.height
        self.width = Config.width

        self.images = {'l':[], 'r':[]}
        self.depths = {'l':[], 'r':[]}

        def __len__(self):

            return len(self.filenames)
        
        def __getitem__(self, idx):
             
            img = {'l':}
        
        def get_data_from_dir(self):

                depth_folder = '../dataset/data_depth_annotated/{}'.format('train' if is_train else 'val')

                for sync in os.listdir(depth_folder):

                    date = sync[:10]
                         
                    depth_folder = os.path.join(sync, 'proj_depth/ground_truth/image_0{}')

                    for img_num in os.listdir(os.path.join(depth_folder, depth_folder(2))):
                            
                        full_img_path = os.path.join(self.dataset_path, date, sync, 'image_0{}/data', '{:010d}'.fomrat(img_num))
                        full_depth_path = os.path.join(depth_folder, sync, depth_folder, img_num)

                        l_img = pil.

                        self.images['l'].append(full_img_path(2))
                        self.images['r'].append(full_img_path(3))

                        l_depth = 

                        self.depths['l'].append(full_depth_path(2))
                        self.depths['r'].append(full_depth_path(3))