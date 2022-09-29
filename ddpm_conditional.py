import argparse
import os
import copy
import numpy as np
import torch
import torch.nn as nn
from types import SimpleNamespace
from tqdm import tqdm
from torch import optim
from utils import *
from modules import UNet_conditional, EMA
import logging
from torch.utils.tensorboard import SummaryWriter

import wandb

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, num_classes=10, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.model = UNet_conditional(num_classes=num_classes).to(device)
        self.ema_model = copy.deepcopy(self.model).eval().requires_grad_(False)
        self.device = device

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))
    
    @torch.inference_mode()
    def sample(self, use_ema, n, labels, cfg_scale=3):
        logging.info(f"Sampling {n} new images....")
        model = self.ema_model if use_ema else self.model
        model.eval()
        with torch.inference_mode():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = self.model(x, t, labels)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x


    def one_epoch(self, train=True, use_wandb=False):
        avg_loss = 0.
        if train: self.model.train()
        else: self.model.eval()
        pbar = tqdm(self.train_dataloader)
        for i, (images, labels) in enumerate(pbar):
            with torch.autocast("cuda") and (torch.inference_mode() if not train else torch.enable_grad()):
                images = images.to(self.device)
                labels = labels.to(self.device)
                t = self.sample_timesteps(images.shape[0]).to(self.device)
                x_t, noise = self.noise_images(images, t)
                if np.random.random() < 0.1:
                    labels = None
                predicted_noise = self.model(x_t, t, labels)
                loss = self.mse(noise, predicted_noise)
                avg_loss += loss
                if train:
                    self.optimizer.zero_grad()
                    self.scheduler.step()
                    loss.backward()
                    self.optimizer.step()
                    self.ema.step_ema(self.ema_model, self.model)

                pbar.set_postfix(MSE=loss.item())
                if use_wandb and train:
                    wandb.log({"train_mse": loss.item(),
                               "learning_rate": self.scheduler.get_last_lr()[0]})
        
        return avg_loss.mean().item()

    @torch.inference_mode()
    def log_images(self, run_name, epoch, use_wandb=False):
        labels = torch.arange(10).long().to(self.device)
        sampled_images = self.sample(use_ema=False, n=len(labels), labels=labels)
        ema_sampled_images = self.sample(use_ema=True, n=len(labels), labels=labels)
        plot_images(sampled_images)
        save_images(sampled_images, os.path.join("results", run_name, f"{epoch}.jpg"))
        save_images(ema_sampled_images, os.path.join("results", run_name, f"{epoch}_ema.jpg"))
        torch.save(self.model.state_dict(), os.path.join("models", run_name, f"ckpt.pt"))
        torch.save(self.ema_model.state_dict(), os.path.join("models", run_name, f"ema_ckpt.pt"))
        torch.save(self.optimizer.state_dict(), os.path.join("models", run_name, f"optim.pt"))
        if use_wandb:
            wandb.log({"sampled_images": [wandb.Image(img.permute(1,2,0).cpu().numpy()) for img in sampled_images]})
            wandb.log({"ema_sampled_images": [wandb.Image(img.permute(1,2,0).cpu().numpy()) for img in ema_sampled_images]})
            at = wandb.Artifact("model", type="model", metadata={"epoch": epoch})
            at.add_dir(os.path.join("models", run_name))
            wandb.log_artifact(at)

    def fit(self, args):
        setup_logging(args.run_name)
        device = args.device
        self.train_dataloader, self.val_dataloader = get_data(args)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=0.001)
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=args.lr, 
                                                 steps_per_epoch=len(self.train_dataloader), epochs=args.epochs)
        self.mse = nn.MSELoss()
        self.ema = EMA(0.995)

        for epoch in range(args.epochs):
            self.model.train()
            logging.info(f"Starting epoch {epoch}:")
            _  = self.one_epoch(train=True, use_wandb=args.use_wandb)
            
            ## validation
            avg_loss = self.one_epoch(train=False, use_wandb=args.use_wandb)
            if args.use_wandb:
                wandb.log({"val_mse": avg_loss})
            if epoch % 5 == 0:
                self.log_images(run_name=args.run_name, epoch=epoch, use_wandb=args.use_wandb)




defaults = SimpleNamespace(    
    run_name = "DDPM_conditional",
    epochs = 100,
    noise_steps=200,
    seed = 42,
    batch_size = 10,
    img_size = 64,
    num_classes = 10,
    dataset_path = get_cifar(img_size=64),
    train_folder = "train",
    val_folder = "test",
    device = "cuda",
    slice_size = 1,
    use_wandb = True,
    fp16 = True,
    lr = 3e-4)


def parse_args():
    argparser = argparse.ArgumentParser(description='Process hyper-parameters')
    argparser.add_argument('--run_name', type=str, default=defaults.run_name, help='name of the run')
    argparser.add_argument('--epochs', type=int, default=defaults.epochs, help='number of epochs')
    argparser.add_argument('--seed', type=int, default=defaults.seed, help='random seed')
    argparser.add_argument('--batch_size', type=int, default=defaults.batch_size, help='batch size')
    argparser.add_argument('--img_size', type=int, default=defaults.img_size, help='image size')
    argparser.add_argument('--num_classes', type=int, default=defaults.num_classes, help='number of classes')
    argparser.add_argument('--dataset_path', type=str, default=defaults.dataset_path, help='path to dataset')
    argparser.add_argument('--device', type=str, default=defaults.device, help='device')
    argparse.add_argument('--use_wandb', type=bool, default=defaults.use_wandb, help='use wandb')
    argparser.add_argument('--lr', type=float, default=defaults.lr, help='learning rate')
    return argparser.parse_args()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    train(args)
