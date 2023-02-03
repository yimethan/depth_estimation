from albumentations import Compose, BboxParams, \
    RandomBrightnessContrast, GaussNoise, RGBShift, CLAHE, RandomGamma, HorizontalFlip, RandomResizedCrop


class Transform(object):
    def __init__(self, box_format='coco'):

        # The `coco` format
        # `[x_min, y_min, width, height]`, e.g. [97, 12, 150, 200].
        # The `pascal_voc` format
        # `[x_min, y_min, x_max, y_max]`, e.g. [97, 12, 247, 212].
        # The `albumentations` format
        # is like `pascal_voc`, but normalized,
        # in other words: `[x_min, y_min, x_max, y_max]`, e.g. [0.2, 0.3, 0.4, 0.5].
        # The `yolo` format
        # `[x, y, width, height]`, e.g. [0.1, 0.2, 0.3, 0.4];
        # `x`, `y` - normalized bbox center; `width`, `height` - normalized bbox width and height.
        
        self.tsfm = Compose([
            HorizontalFlip(),
            # RandomResizedCrop(512, 512, scale=(0.75, 1)),
            RandomBrightnessContrast(0.4, 0.4),
            GaussNoise(),
            RGBShift(),
            CLAHE(),
            RandomGamma()
        ], bbox_params=BboxParams(format=box_format, min_visibility=0.75, label_fields=['labels']))

    def __call__(self, img, boxes, labels):
        augmented = self.tsfm(image=img, bboxes=boxes, labels=labels)
        img, boxes = augmented['image'], augmented['bboxes']
        return img, boxes