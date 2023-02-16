from dataset import kitti_dataset
import config.config
from loss import loss
from torch.utils.data import DataLoader
from model.model import GcNet
from trainer import Trainer
import torch
import os

def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines

def train(config):

    datasets_dict = {"kitti": kitti_dataset.KITTIRAWDataset,
                    "kitti_odom": kitti_dataset.KITTIOdomDataset}
    dataset = datasets_dict['kitti']

    fpath = os.path.join(os.path.dirname(config.root), "splits", config.split, "{}_files.txt")

    train_filenames = readlines(fpath.format("train"))
    val_filenames = readlines(fpath.format("val"))
    img_ext = '.png' if config.png else '.jpg'

    train_ds = kitti_dataset.KITTIDataset(config.root, train_filenames, config.resize_size[0], config.resize_size[1],
                                          config.frame_ids, 4, is_train=True, img_ext=img_ext)
    train_dl = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=config.num_workers,
                          collate_fn=train_ds.collate_fn, pin_memory=True)

    val_ds = kitti_dataset.KITTIDataset(config.root, val_filenames, config.resize_size[0], config.resize_size[1],
                                          config.frame_ids, 4, is_train=False, img_ext=img_ext)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=config.num_workers,
                        collate_fn=val_ds.collate_fn, pin_memory=True)
    
    num_train_samples = len(train_filenames)
    num_total_steps = num_train_samples // config.batch_size * config.epochs

    if config.gpu:
        model = model.cuda()
        
    loss_func = loss.loss(config)

    # TODO: load model to train (for depth estimation)
    model = GcNet(config.resize_size[0], config.resize_size[1], config.max_disp)

    epochs = config.epochs
    config.max_iter = len(train_dl) * epochs
    config.steps = (int(config.max_iter * 0.6), int(config.max_iter * 0.8))

    trainer = Trainer(config, model, loss_func, train_dl, val_dl)
    trainer.train()