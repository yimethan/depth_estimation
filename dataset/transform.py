from torchvision import transforms
from config.config import Config
import numpy as np
from PIL import Image

class Transform():
    
    def __init__(self):

        self.data_transform = transforms.Compose([
            transforms.Resize(size=(Config.height, Config.width)),
            transforms.ToTensor()
        ])

    def __call__(self, img):

        img = self.data_transform(img)

        return img