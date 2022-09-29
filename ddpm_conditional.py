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
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
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

    def sample(self, model, n, labels, cfg_scale=3):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, labels)
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
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x


def train(args):
    setup_logging(args.run_name)
    device = args.device
    train_dataloader, val_dataloader = get_data(args)
    model = UNet_conditional(num_classes=args.num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(noise_steps=args.noise_steps, img_size=args.img_size, device=device)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(train_dataloader)
    ema = EMA(0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(train_dataloader)
        for i, (images, labels) in enumerate(pbar):
            with torch.autocast("cuda"):
                images = images.to(device)
                labels = labels.to(device)
                t = diffusion.sample_timesteps(images.shape[0]).to(device)
                x_t, noise = diffusion.noise_images(images, t)
                if np.random.random() < 0.1:
                    labels = None
                predicted_noise = model(x_t, t, labels)
                loss = mse(noise, predicted_noise)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                ema.step_ema(ema_model, model)

                pbar.set_postfix(MSE=loss.item())
                if args.use_wandb:
                    wandb.log({"MSE": loss.item()})
                logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        if epoch % 1 == 0:
            labels = torch.arange(10).long().to(device)
            sampled_images = diffusion.sample(model, n=len(labels), labels=labels)
            ema_sampled_images = diffusion.sample(ema_model, n=len(labels), labels=labels)
            if args.use_wandb:
                wandb.log({"sampled_images": [wandb.Image(img.permute(1,2,0).cpu().numpy()) for img in sampled_images]})
                wandb.log({"ema_sampled_images": [wandb.Image(img.permute(1,2,0).cpu().numpy()) for img in ema_sampled_images]})
            plot_images(sampled_images)
            save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
            save_images(ema_sampled_images, os.path.join("results", args.run_name, f"{epoch}_ema.jpg"))
            torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))
            torch.save(ema_model.state_dict(), os.path.join("models", args.run_name, f"ema_ckpt.pt"))
            torch.save(optimizer.state_dict(), os.path.join("models", args.run_name, f"optim.pt"))
            if args.use_wandb:
                at = wandb.Artifact("model", type="model", metadata={"epoch": epoch})
                at.add_dir(os.path.join("models", args.run_name))
                wandb.log_artifact(at)


defaults = SimpleNamespace(    
    run_name = "DDPM_conditional",
    epochs = 20,
    noise_steps=200,
    seed = 42,
    batch_size = 10,
    img_size = 64,
    num_classes = 10,
    dataset_path = get_cifar(img_size=64),
    train_folder = "train",
    val_folder = "test",
    device = "cuda",
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
