import os
import time
import torch
import torch.nn as nn
import torchvision

from torch.utils.data import DataLoader
from .models import DeepJointFilterModel
from .dataset import Dataset
from .metrics import PSNR
from .utils import Progbar, print_fun, create_dir, imsave, stitch_images
from .htmls import HTML


class DeepJointFilter(object):
    def __init__(self, config):
        self.config = config
        name = config.name + time.strftime("_%y_%d_%H_%M")

        self.train_dataset = Dataset(config)
        self.val_dataset = Dataset(config)
        self.test_dataset = Dataset(config)
        self.sample_iterator = self.val_dataset.create_iterator(config.sample_size)
        self.model = DeepJointFilterModel(config).to(config.device)
        self.psnr = PSNR(255.0).to(config.device)
        self.html = config.html

        self.log_file = os.path.join(config.config_path, "output", name, "log_" + name + ".txt")
        self.save_path = os.path.join(config.config_path, "output", name, config.save_path)
        self.samples_path = os.path.join(config.config_path, "output", name, config.samples_path)
        self.results_path = os.path.join(config.config_path, "output", name, config.outputs_path)
        create_dir(self.save_path)
        create_dir(self.samples_path)
        create_dir(self.results_path)


    def load(self):
        self.model.load()


    def save(self):
        self.model.save()


    def train(self):
        train_loader = DataLoader(
            self.train_dataset,
            self.config.trainer.batch_size,
            num_workers=2,
            drop_last=True,
            shuffle=True
        )

        epoch = 0
        keep_training = True
        max_iteration = int(float(self.config.trainer.max_iters))
        total = len(self.train_dataset)

        while(keep_training):
            epoch += 1
            print('\n\nTraining epoch: %d' % epoch)

            progbar = Progbar(total, width=20, stateful_metrics=['epoch', 'iter'])

            for items in train_loader:
                self.model.train()

                # train
                target, guide, gt = self.cuda(*items)
                output, loss, logs = self.model.process(target, guide, gt)

                self.model.backward(loss)

                # import ipdb; ipdb.set_trace()
                # metrics
                psnr = self.psnr(gt, output)
                mae = (torch.sum(torch.abs(gt - target)) / torch.sum(gt)).float()
                logs.append(('psnr', psnr.item()))
                logs.append(('mae', mae.item()))

                iteration = self.model.iteration


                if iteration >= max_iteration:
                    keep_training = False
                    break

                # logs
                logs = [
                    ("epoch", epoch),
                    ("iter", iteration),
                ] + logs

                progbar.add(len(gt), values=logs if self.config.verbose else [x for x in logs if not x[0].startswith('l_')])


                # log model at checkpoints
                if self.config.log_interval and iteration % self.config.log_interval == 0:
                    self.log(logs)

                # sample model at checkpoints
                if self.config.sample_interval and iteration % self.config.sample_interval == 0:
                    self.sample()

                # evaluate model at checkpoints
                if self.config.eval_interval and iteration % self.config.eval_interval == 0:
                    print('\nstart eval...\n')
                    self.evaluate()
                    print('\nend eval...\n')


        
        print('\nEnd training....')




    def evaluate(self):
        val_loader = DataLoader(
            self.val_dataset,
            self.config.trainer.batch_size,
            num_workers=1,
            drop_last=True,
            shuffle=True
        )

        total = len(self.val_dataset)

        self.model.eval()

        progbar = Progbar(total, width=20, stateful_metrics=['it'])
        iteration = 0

        with torch.no_grad():
            for items in val_loader:
                iteration += 1

                target, guide, gt = self.cuda(*items)
                output, loss, logs = self.model.process(target, guide, gt)


                # metrics
                psnr = self.psnr(gt, output)
                mae = (torch.sum(torch.abs(gt - target)) / torch.sum(gt)).float()
                logs.append(('psnr', psnr.item()))
                logs.append(('mae', mae.item()))


                logs = [("it", iteration), ] + logs
                progbar.add(len(target), values=logs)


    def test(self):
        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
        )

        self.model.eval()


    def sample(self):
        self.model.eval()

        with torch.no_grad():
            iteration = self.model.iteration

            items = next(self.sample_iterator)
            target, guide, gt = self.cuda(*items)
            output, loss, logs = self.model.process(target, guide, gt)


            image_per_row = 2
            if self.config.sample_size <= 6:
                image_per_row = 1

            sample_image = stitch_images(
                self.postprocess(target),
                self.postprocess(guide),
                self.postprocess(output),
                self.postprocess(gt),
                img_per_row = image_per_row
            )

            name = os.path.join(self.samples_path, str(iteration).zfill(7) + ".png")
            print('\nsaving sample ' + name)
            sample_image.save(name)


    def log(self, logs):
        with open(self.log_file, 'a') as f:
            f.write('%s\n' % ' '.join([str(item) for item in logs]))


    def cuda(self, *args):
        return (item.to(self.config.device) for item in args)


    def postprocess(self, img, to_byte=True):
        img = img.permute(0, 2, 3, 1)
        if to_byte:
            # [0, 1] => [0, 255]
            img = img * 255.0
            img = img.byte()
        return img
