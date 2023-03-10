import PIL.Image
from torchvision import transforms
from config.config import Config
import numpy as np
from PIL import Image

class Transform():
    
    def __init__(self):

        self.data_transform = transforms.Compose([
            transforms.Resize(size=(Config.detect_height, Config.detect_width), interpolation=Image.ANTIALIAS),
            transforms.ToTensor()
        ])

    def __call__(self, img):

        # img = Image.fromarray(img)
        img = self.data_transform(img)

        return img