import torch
import torch.optim as optim
import os
import time
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from dataset.load_dataset import Dataset

class Train:

    def __init__(self, config):

        self.config = config

        assert self.config.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.config.width % 32 == 0, "'width' must be a multiple of 32"

        self.device = torch.device("cpu" if self.config.no_cuda else "cuda")

        self.model = Model()

        self.model_optimizer = optim.Adam(self.model.parameters(), self.config.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(self.model_optimizer, self.config.scheduler_step_size, 0.1)

        fpath = os.path.join("../dataset/splits", self.config.split, "{}_files.txt")

        train_filenames = self.readlines(fpath.format("train"))
        val_filenames = self.readlines(fpath.format("val"))
        
        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.config.batch_size * self.config.num_epochs

        train_dataset = Dataset(train_filenames, is_train=True)
        self.train_loader = DataLoader(train_dataset, self.config.batch_size, True,
            num_workers=self.config.num_workers, pin_memory=True, drop_last=True)
        
        val_dataset = Dataset(val_filenames, is_train=False)
        self.val_loader = DataLoader(val_dataset, self.config.batch_size, True,
            num_workers=self.config.num_workers, pin_memory=True, drop_last=True)
        
        self.val_iter = iter(self.val_loader)

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.config.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))
        
    def train(self):

        self.start_time = time.time()

        for epoch in range(self.config.epochs):

            self.run_epoch()

            if (self.epoch + 1) % self.config.save_f == 0:

                save_folder = os.path.join(self.config.log_dir, 'models', 'weights_{}'.format(epoch))
                
                if not os.path.exists(save_folder):

                    os.makedirs(save_folder)

                save_path = os.path.join(save_folder, '{}.pth'.format(self.config.model_name))
                to_save = self.model.state_dict()

                torch.save(to_save, save_path)

                save_path = os.path.join(save_folder, '{}.pth'.format('optimizer'))
                torch.save(self.model_optimizer.state_dict(), save_path)

    def run_epoch(self):

            self.model.train()

            for batch_idx, inputs in enumerate(self.train_loader):

                start_time = time.time()

                # outputs, losses = self.model(inputs)

                self.model_optimizer.zero_grad()
                # losses["loss"].backward()
                self.model_optimizer.step()

                duration = time.time() - start_time

                # log less frequently after the first 2000 steps to save time & disk space
                early_phase = batch_idx % self.config.log_frequency == 0 and self.step < 2000
                late_phase = self.step % 2000 == 0

                if early_phase or late_phase:
                    # self.log_time(batch_idx, duration, losses["loss"].cpu().data)

                    # if "depth_gt" in inputs:
                    #     self.compute_depth_losses(inputs, outputs, losses)

                    # self.log("train", inputs, outputs, losses)
                    self.val()

                self.step += 1

            self.model_lr_scheduler.step()
        
    def readlines(filename):

        with open(filename, 'r') as f:

            lines = f.read().splitlines()

        return lines
    
    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.config.batch_size / duration
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
    
    # def log(self, mode, inputs, outputs, losses):
    #     """Write an event to the tensorboard events file
    #     """
    #     writer = self.writers[mode]
    #     for l, v in losses.items():
    #         writer.add_scalar("{}".format(l), v, self.step)

    #     for j in range(min(4, self.config.batch_size)):  # write a maxmimum of four images
    #         for s in self.config.scales:
    #             for frame_id in self.config.frame_ids:
    #                 writer.add_image(
    #                     "color_{}_{}/{}".format(frame_id, s, j),
    #                     inputs[("color", frame_id, s)][j].data, self.step)
    #                 if s == 0 and frame_id != 0:
    #                     writer.add_image(
    #                         "color_pred_{}_{}/{}".format(frame_id, s, j),
    #                         outputs[("color", frame_id, s)][j].data, self.step)

    #             writer.add_image("disp_{}/{}".format(s, j),
    #                 self.normalize_image(outputs[("disp", s)][j]), self.step)

    #             if self.config.predictive_mask:
    #                 for f_idx, frame_id in enumerate(self.config.frame_ids[1:]):
    #                     writer.add_image(
    #                         "predictive_mask_{}_{}/{}".format(frame_id, s, j),
    #                         outputs["predictive_mask"][("disp", s)][j, f_idx][None, ...],
    #                         self.step)

    #             elif not self.config.disable_automasking:
    #                 writer.add_image(
    #                     "automask_{}/{}".format(s, j),
    #                     outputs["identity_selection/{}".format(s)][j][None, ...], self.step)
                    
    def normalize_image(x):
        """Rescale image pixels to span range [0, 1]
        """
        ma = float(x.max().cpu().data)
        mi = float(x.min().cpu().data)
        d = ma - mi if ma != mi else 1e5
        return (x - mi) / d