from trainer.trainer import Trainer
from dataset.kitti import KITTIDataset
from config.kitti import Config
from loss.loss import Loss
from torch.utils.data import DataLoader
from model.model import Model, DlaNet

def train(cfg):
    train_ds = KITTIDataset(cfg.root, mode=cfg.split, resize_size=cfg.resize_size)
    train_dl = DataLoader(train_ds, batch_size=1, shuffle=True,
                          num_workers=cfg.num_workers, collate_fn=train_ds.collate_fn, pin_memory=True)

    detect_model = DlaNet(34)
    detect_model.eval()

    model = Model(cfg)
    
    if cfg.gpu:
        model = model.cuda()
    loss_func = Loss(cfg)

    epoch = 100
    cfg.max_iter = len(train_dl) * epoch
    cfg.steps = (int(cfg.max_iter * 0.6), int(cfg.max_iter * 0.8))

    trainer = Trainer(cfg, model, loss_func, train_dl, None)
    trainer.train()

if __name__ == '__main__':
    cfg = Config
    train(cfg)