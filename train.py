import torch
import torch.optim as optim
import torch.functional as F
import os
import time
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from dataset.load_dataset import Dataset
from config.config import Config
from model.model import Model

class Train:

    def __init__(self, config):

        assert Config.height % 32 == 0, "'height' must be a multiple of 32"
        assert Config.width % 32 == 0, "'width' must be a multiple of 32"

        self.device = torch.device("cpu" if Config.no_cuda else "cuda")

        self.model = Model()

        self.model_optimizer = optim.Adam(self.model.parameters(), Config.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(self.model_optimizer, Config.scheduler_step_size, 0.1)

        train_dataset = Dataset(is_train=True)
        self.train_loader = DataLoader(train_dataset, Config.batch_size, True,
            num_workers=Config.num_workers, pin_memory=True, drop_last=True)
        val_dataset = Dataset(is_train=False)
        self.val_loader = DataLoader(val_dataset, Config.batch_size, True,
            num_workers=Config.num_workers, pin_memory=True, drop_last=True)

        assert len(train_dataset.images['l']) == len(train_dataset.images['r']), \
            "Number of left and right images of train set must be equal"
        assert len(train_dataset.images['l']) == len(train_dataset.depth['l']), \
            "Number of left images and depth gt of train set must be equal"
        assert len(train_dataset.images['r']) == len(train_dataset.depth['r']), \
            "Number of right images and depth gt of train set must be equal"

        assert len(val_dataset.images['l']) == len(val_dataset.images['r']), \
            "Number of left and right images of val set must be equal"
        assert len(val_dataset.images['l']) == len(val_dataset.depth['l']), \
            "Number of left images and depth gt of val set must be equal"
        assert len(val_dataset.images['r']) == len(val_dataset.depth['r']), \
            "Number of right images and depth gt of val set must be equal"

        num_train_samples = len(train_dataset)
        self.num_total_steps = num_train_samples // Config.batch_size * Config.epochs
        
        self.val_iter = iter(self.val_loader)

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", Config.split)
        print("There are {:d} training pairs and {:d} validation pairs\n".format(
            len(train_dataset), len(val_dataset)))
        
    def train(self):

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
                save_path = os.path.join(save_folder, '{}.pth'.format(Config.model_name))
                torch.save(self.model.state_dict(), save_path)

                # save optimizer
                save_path = os.path.join(save_folder, '{}.pth'.format('optimizer'))
                torch.save(self.model_optimizer.state_dict(), save_path)

    def run_epoch(self):

        self.model.train()

        for batch_idx, inputs in enumerate(self.train_loader):

            start_time = time.time()

            # output = {'l_prob':left_prob, 'r_prob':right_prob,
            #     'l_newinp':l_newinp, 'r_newinp':r_newinp,
            #     'l_bbox':left_y, 'r_bbox':right_y}
            outputs = self.model(inputs['l_img'], inputs['r_img'])

            loss, car_loss = self.compute_loss(inputs['l_img'], inputs['r_img'],
                                    outputs['l_prob'], outputs['r_prob'])
            self.model_optimizer.zero_grad()
            loss.backward()
            self.model_optimizer.step()

            duration = time.time() - start_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % Config.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                self.log_time(batch_idx, duration, loss.cpu().data)

                errors = self.compute_errors(inputs, outputs)
                car_errors = self.compute_car_errors(inputs, outputs)

                self.log("train", inputs, outputs, loss, car_errors, errors)
                self.val()

            self.step += 1

        self.model_lr_scheduler.step()

        return

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
            # output = {'l_prob':left_prob, 'r_prob':right_prob,
            #     'l_newinp':l_newinp, 'r_newinp':r_newinp,
            #     'l_bbox':left_y, 'r_bbox':right_y}
            outputs = self.model(inputs['l_img'], inputs['r_img'])

            loss = self.compute_loss(inputs['l_img'], inputs['r_img'],
                                    outputs['l_prob'], outputs['r_prob'])
            
            self.model_optimizer.zero_grad()
            loss.backward()
            self.model_optimizer.step()

            errors = self.compute_errors(inputs, outputs)
            car_errors = self.compute_car_errors(inputs, outputs)

            self.log("val", inputs, outputs, loss, car_errors, errors)
            del inputs, outputs, losses

        self.set_train()
        
    def readlines(filename):

        with open(filename, 'r') as f:

            lines = f.read().splitlines()

        return lines
    
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

    def sec_to_hm_str(t):
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
    
    def log(self, mode, inputs, outputs, loss, car_loss, errors):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in loss.items():
            # loss
            writer.add_scalar("loss_{}".format(l), v, self.step)

        for l, v in car_loss.items():
            # car_loss
            writer.add_scalar('car_loss_{}'.format(l), v, self.step)

        # errors
        for side in [2, 3]:
            writer.add_scalar('errors_{}/abs_rel/{}'.format(side, errors['left' if side == 2 else 'right']['abs_rel']), self.step)
            writer.add_scalar('errors_{}/sq_rel/{}'.format(side, errors['left' if side == 2 else 'right']['sq_rel']), self.step)
            writer.add_scalar('errors_{}/rmse/{}'.format(side, errors['left' if side == 2 else 'right']['rmse']), self.step)
            writer.add_scalar('errors_{}/rmse_log/{}'.format(side, errors['left' if side == 2 else 'right']['rmse_log']), self.step)
            writer.add_scalar('errors_{}/a1/{}'.format(side, errors['left' if side == 2 else 'right']['a1']), self.step)
            writer.add_scalar('errors_{}/a2/{}'.format(side, errors['left' if side == 2 else 'right']['a2']), self.step)
            writer.add_scalar('errors_{}/a3/{}'.format(side, errors['left' if side == 2 else 'right']['a3']), self.step)

        for j in range(min(4, Config.batch_size)):  # write a maxmimum of four images

            # original images
            writer.add_image("orig_img_2/{}".format(j), inputs['l_img'], self.step)
            writer.add_image("orig_img_3/{}".format(j), inputs['r_img'], self.step)
                
            # new images with only vehicles
            writer.add_image("car_img_2/{}".format(j), outputs['l_newinp'], self.step)
            writer.add_image("car_img_3/{}".format(j), outputs['r_newinp'], self.step)

            # depth map
            writer.add_image("disp_2/{}".format(j),
                self.normalize_image(outputs['l_prob'][j]), self.step)
            writer.add_image("disp_3/{}".format(j),
                self.normalize_image(outputs['r_prob'][j]), self.step)
                    
    def normalize_image(x):
        """Rescale image pixels to span range [0, 1]
        """
        ma = float(x.max().cpu().data)
        mi = float(x.min().cpu().data)
        d = ma - mi if ma != mi else 1e5
        return (x - mi) / d
    
    def compute_car_errors(self, inputs, outputs):
        
        # l_pred -> l_newpred / r_pred -> r_newpred
        # l_gt -> l_newgt / r_gt r_newgt

        outputs['l_pred'] = Model.generate_newinp(outputs['l_pred'], outputs['l_bbox'])
        outputs['r_pred'] = Model.generate_newinp(outputs['r_pred'], outputs['r_bbox'])
        inputs['l_depth'] = Model.generate_newinp(inputs['l_depth'], outputs['l_bbox'])
        inputs['r_depth'] = Model.generate_newinp(inputs['r_depth'], outputs['r_bbox'])

        return self.compute_errors(inputs, outputs)
    
    def compute_loss(self, l_gt, r_gt, l_pred, r_pred):

        loss_left = torch.mean((l_pred - l_gt)**2)
        loss_right = torch.mean((r_pred, r_gt)**2)

        return loss_left + loss_right
    
    def compute_errors(self, inputs, outputs):

        errors = {'left':{}, 'right':{}}

        l_pred = outputs[("l_pred", 0, 0)]
        r_pred = outputs[('r_pred', 0, 0)]

        l_pred = torch.clamp(F.interpolate(
            l_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        l_pred = l_pred.detach()
        r_pred = torch.clamp(F.interpolate(
            r_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        r_pred = r_pred.detach()

        l_gt = inputs["l_depth"]
        l_mask = l_gt > 0
        
        r_gt = inputs["r_depth"]
        r_mask = r_gt > 0

        # garg/eigen crop
        l_crop_mask = torch.zeros_like(l_mask)
        l_crop_mask[:, :, 153:371, 44:1197] = 1
        l_mask = l_mask * l_crop_mask

        r_crop_mask = torch.zeros_like(r_mask)
        r_crop_mask[:, :, 153:371, 44:1197] = 1
        r_mask = r_mask * r_crop_mask

        l_gt = l_gt[l_mask]
        l_pred = l_pred[l_mask]
        l_pred *= torch.median(l_gt) / torch.median(l_pred)

        r_gt = r_gt[r_mask]
        r_pred = r_pred[r_mask]
        r_pred *= torch.median(r_gt) / torch.median(r_pred)

        l_pred = torch.clamp(l_pred, min=1e-3, max=80)
        r_pred = torch.clamp(r_pred, min=1e-3, max=80)

        errors['left'] = self.compute_depth_errors(l_gt, l_pred)
        errors['right'] = self.compute_depth_errors(r_gt, r_pred)

        return errors


    def compute_depth_errors(gt, pred):
        """Computation of error metrics between predicted and ground truth depths
        """
        thresh = torch.max((gt / pred), (pred / gt))
        a1 = (thresh < 1.25     ).float().mean()
        a2 = (thresh < 1.25 ** 2).float().mean()
        a3 = (thresh < 1.25 ** 3).float().mean()

        rmse = (gt - pred) ** 2
        rmse = torch.sqrt(rmse.mean())

        rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
        rmse_log = torch.sqrt(rmse_log.mean())

        abs_rel = torch.mean(torch.abs(gt - pred) / gt)

        sq_rel = torch.mean((gt - pred) ** 2 / gt)

        errors = {'abs_rel':abs_rel, 'sq_rel':sq_rel, 'rmse':rmse,
                'rmse_log':rmse_log, 'a1': a1, 'a2': a2, 'a3':a3}

        return errors
