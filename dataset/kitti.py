import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from dataset.utils import gaussian_radius, draw_umich_gaussian
import os
import cv2
import xml.etree.ElementTree as ET
from dataset.transform import Transform

opj = os.path.join

class KITTIDataset(Dataset):

    CLASSES_NAME = (
        'car', 'van', 'truck', 'tram'
    )

    def __init__(self, root='../dataset/kitti', resize_size=(800, 1024), mode='train',
                 mean=(0.40789654, 0.44719302, 0.47026115), std=(0.28863828, 0.27408164, 0.27809835)):

        self.root = root
        self.image_folder = opj(self.root, '')
        self.annotation = opj(self.root, '')
        self.mode = mode
        
        self.transform = Transform('pascal_voc') # [x_min, y_min, x_max, y_max]
        self.resize_size = resize_size
        self.mean = mean
        self.std = std
        self.down_stride = 4

        self.category2id = {k: v for v, k in enumerate(self.CLASSES_NAME)}
        self.id2category = {v: k for k, v in self.category2id.items()}

        with open(opj(self.root, '')) as f:

            self.samples = f.read().split()

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        sample = self.samples[index]
        img = Image.open(opj(self.image_folder, f'{sample}.png'))
        label = self.parse_annotation(opj(self.annotation, f'{sample}.xml'))
        img = np.array(img)
        raw_h, raw_w, _ = img.shape
        info = {'raw_height': raw_h, 'raw_width': raw_w}
        if self.mode == 'train':
            img, boxes = self.transform(img, label['boxes'], label['labels'])
        else:
            boxes = np.array(label['boxes'])
        boxes_w, boxes_h = boxes[..., 2] - boxes[..., 0], boxes[..., 3] - boxes[..., 1]
        ct = np.array([(boxes[..., 0] + boxes[..., 2]) / 2,
                       (boxes[..., 1] + boxes[..., 3]) / 2], dtype=np.float32).T

        img, boxes = self.preprocess_img_boxes(img, self.resize_size, boxes)
        info['resize_height'], info['resize_width'] = img.shape[:2]

        return super().__getitem__(index)