import torch
import torch.optim as optim
import torch.nn.functional as F
import os
import time
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
import kornia
from tqdm import tqdm

from dataset.load_dataset import Dataset
from config.config import Config
from model.model import Model

class Train:

    def __init__(self):

        assert Config.height % 32 == 0, "'height' must be a multiple of 32"
        assert Config.width % 32 == 0, "'width' must be a multiple of 32"

        self.device = torch.device("cuda")

        self.model = Model()

        self.model_optimizer = optim.Adam(self.model.parameters(), Config.lr)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(self.model_optimizer, Config.scheduler_f, 0.1)

        print("Loading dataset...")
        train_dataset = Dataset(is_train=True)
        self.train_loader = DataLoader(train_dataset, Config.batch_size, True,
            num_workers=Config.num_workers, pin_memory=True, drop_last=True)
        val_dataset = Dataset(is_train=False)
        self.val_loader = DataLoader(val_dataset, Config.batch_size, True,
            num_workers=Config.num_workers, pin_memory=True, drop_last=True)

        num_train_samples = len(train_dataset)
        self.num_total_steps = num_train_samples // Config.batch_size * Config.epochs
        
        self.val_iter = iter(self.val_loader)

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(Config.log_dir, mode))

        # self.depth_metric_names = [
        #     "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]
        
    def train(self):

        print("Start training")

        self.start_time = time.time()

        self.model.train()

        self.step = 0
        self.epoch = 0

        for self.epoch in range(Config.epochs):

            self.run_epoch()

            if (self.epoch + 1) % Config.save_f == 0:

                save_folder = os.path.join(Config.log_dir, 'models', 'weights_epoch{}'.format(self.epoch))
                
                if not os.path.exists(save_folder):

                    os.makedirs(save_folder)

                # save model
                print("saving model for this epoch")
                save_path = os.path.join(save_folder, '{}.pth'.format(Config.model_name))
                torch.save(self.model.state_dict(), save_path)

                # save optimizer
                save_path = os.path.join(save_folder, '{}.pth'.format('optimizer'))
                torch.save(self.model_optimizer.state_dict(), save_path)

    def run_epoch(self):

        self.model.train()

        for batch_idx, inputs in enumerate(self.train_loader):

            start_time = time.time()

            # output = {'l_pred':left_prob, 'r_pred':right_prob,
            #     'l_newinp':l_newinp, 'r_newinp':r_newinp,
            #     'l_bbox':left_y, 'r_bbox':right_y}
            outputs = self.model(inputs)

            loss = self.compute_loss(inputs['l_depth'], outputs['l_pred']) + self.compute_loss(inputs['r_depth'], outputs['r_pred'])
            loss.requires_grad_(True)

            self.model_optimizer.zero_grad()
            loss.backward()
            self.model_optimizer.step()

            duration = time.time() - start_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % Config.log_f == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                self.log_time(batch_idx, duration, loss.cpu().data)

                errors = self.compute_errors(inputs, outputs)
                car_errors = self.compute_car_errors(inputs, outputs)

                self.log("train", inputs, outputs, loss, car_errors, errors)
                self.val()

            self.step += 1

        self.model_lr_scheduler.step()

    def val(self):
        """Validate the model on a single minibatch
        """
        self.model.eval()

        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            # output = {'l_pred':left_prob, 'r_pred':right_prob,
            #     'l_newinp':l_newinp, 'r_newinp':r_newinp,
            #     'l_bbox':left_y, 'r_bbox':right_y}
            outputs = self.model(inputs)

            loss = self.compute_loss(inputs['l_depth'], outputs['l_pred']) + self.compute_loss(inputs['r_depth'], outputs['r_pred'])
            loss.requires_grad_(True)
            
            self.model_optimizer.zero_grad()
            self.model_optimizer.step()

            errors = self.compute_errors(inputs, outputs)
            car_errors = self.compute_car_errors(inputs, outputs)

            self.log("val", inputs, outputs, loss, car_errors, errors)
            del inputs, outputs, loss

        self.model.train()
    
    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = Config.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  self.sec_to_hm_str(time_sofar), self.sec_to_hm_str(training_time_left)))

    def sec_to_hm_str(self, t):
        """Convert time in seconds to a nice string
        e.g. 10239 -> '02h50m39s'
        """
        # h, m, s = self.sec_to_hm(t)
        t = int(t)
        s = t % 60
        t //= 60
        m = t % 60
        t //= 60
        return "{:02d}h{:02d}m{:02d}s".format(t, m, s)
    
    def log(self, mode, inputs, outputs, loss, car_errors, errors):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]

        writer.add_scalar("loss", loss, self.step)

        # errors
        writer.add_scalar('rmse_2', errors[0], self.step)
        writer.add_scalar('rmse_3', errors[1], self.step)

        # car errors
        writer.add_scalar('car_rmse_2', car_errors[0], self.step)
        writer.add_scalar('car_rmse_3', car_errors[1], self.step)

        l_img = torch.squeeze(inputs['l_img'])
        r_img = torch.squeeze(inputs['r_img'])
        writer.add_image("orig_img_2", l_img, self.step)
        writer.add_image("orig_img_3", r_img, self.step)

        # new images with only vehicles
        l_newinp = torch.squeeze(outputs['l_newinp'])
        r_newinp = torch.squeeze(outputs['r_newinp'])
        writer.add_image("car_img_2", l_newinp, self.step)
        writer.add_image("car_img_3", r_newinp, self.step)

        # depth map
        l_pred = outputs['l_pred'].reshape(1, Config.height//2, Config.width//2)
        r_pred = outputs['r_pred'].reshape(1, Config.height//2, Config.width//2)
        writer.add_image("disp_2", self.normalize_image(l_pred), self.step)
        writer.add_image("disp_3", self.normalize_image(r_pred), self.step)
                    
    def normalize_image(self, x):
        """Rescale image pixels to span range [0, 1]
        """
        ma = float(x.max().cpu().data)
        mi = float(x.min().cpu().data)
        d = ma - mi if ma != mi else 1e5
        return (x - mi) / d
    
    def compute_car_errors(self, inputs, outputs):
        
        # l_pred -> l_newpred / r_pred -> r_newpred
        # l_gt -> l_newgt / r_gt r_newgt

        # pred [1, 128, 128] depth [1, 1, 128, 128]

        outputs['l_pred'] = self.model.generate_newpred(outputs['l_pred'], outputs['l_bbox'][:][-1])
        outputs['r_pred'] = self.model.generate_newpred(outputs['r_pred'], outputs['r_bbox'][:][-1])
        inputs['l_depth'] = self.model.generate_newgt(inputs['l_depth'], outputs['l_bbox'][:][-1])
        inputs['r_depth'] = self.model.generate_newgt(inputs['r_depth'], outputs['r_bbox'][:][-1])

        # mewpred [1, 64, 64], newgt [1, 1, 128, 128]

        return self.compute_errors(inputs, outputs)

    def compute_loss(self, target, pred):

        # target [1, 1, 128, 128] pred [1, 128, 128]

        pred = pred.view(1, 1, Config.height, Config.width)

        # Edges
        target_edges = kornia.filters.spatial_gradient(target)
        dx_true = target_edges[:, :, 0]
        dy_true = target_edges[:, :, 1]

        pred_edges = kornia.filters.spatial_gradient(pred)
        dx_pred = target_edges[:, :, 0]
        dy_pred = target_edges[:, :, 1]

        weights_x = torch.exp(torch.mean(torch.abs(dx_true)))
        weights_y = torch.exp(torch.mean(torch.abs(dy_true)))

        # Depth smoothness
        smoothness_x = dx_pred * weights_x
        smoothness_y = dy_pred * weights_y

        depth_smoothness_loss = torch.mean(torch.abs(smoothness_x)) + torch.mean(torch.abs(smoothness_y))

        # Structural similarity (SSIM) index
        ssim_loss = 1 - kornia.losses.ssim_loss(pred, target, window_size=7, reduction='mean')

        # Point-wise depth
        l1_loss = torch.mean(torch.abs(target - pred))

        loss = ((0.75 * ssim_loss) + (0.20 * l1_loss) + (0.15 * depth_smoothness_loss))

        return loss
    
    def compute_errors(self, inputs, outputs):

        errors = {'left', 'right'}

        l_pred = outputs['l_pred'] # [1, 256, 256]
        r_pred = outputs['r_pred']
        l_gt = inputs["l_depth"] # [1, 1, 512, 512]
        r_gt = inputs["r_depth"]

        # TODO: gt size to pred size
        l_gt = transforms.functional.resize(l_gt, (Config.height//2, Config.width//2))
        r_gt = transforms.functional.resize(r_gt, (Config.height//2, Config.width//2))
        l_gt = torch.squeeze(l_gt, 0)
        r_gt = torch.squeeze(r_gt, 0)

        l_pred = torch.clamp(l_pred, min=1e-3, max=Config.maxdisp)
        r_pred = torch.clamp(r_pred, min=1e-3, max=Config.maxdisp)

        l_pred = l_pred.detach()
        r_pred = r_pred.detach()

        left_rmse = self.compute_depth_errors(l_gt, l_pred)
        right_rmse = self.compute_depth_errors(r_gt, r_pred)

        return left_rmse, right_rmse

    def compute_depth_errors(self, gt, pred):
        """Computation of error metrics between predicted and ground truth depths
        """
        # gt [1, 64, 64], pred [1, 128, 128] or [1, 64, 64]

        pred = transforms.functional.resize(pred, (Config.height//2, Config.width//2))

        rmse = (gt - pred) ** 2
        rmse = torch.sqrt(rmse.mean())

        return rmse


if __name__ == '__main__':

    train = Train()
    train.train()