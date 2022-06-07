import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from utils import get_data
from modules import UNet
import logging
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule()
        self.alpha = 1 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        self.img_size = img_size
        self.device = device

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None].to(self.device)
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None].to(self.device)
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        # return torch.clamp((torch.sqrt(torch.rand(n))*1000).long(), 1, 999)
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n, start_x=None, start_t=None):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            if start_x is None:
                x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            else:
                x = start_x
            for i in tqdm(reversed(range(1, self.noise_steps if start_t is None else start_t)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None].to(self.device)
                alpha_hat = self.alpha_hat[t][:, None, None, None].to(self.device)
                beta = self.beta[t][:, None, None, None].to(self.device)
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + beta * noise
        model.train()
        return x

    def sample_from_timestep(self, model, image, timesteps):
        denoised_images = []
        noised_images = []
        for t in timesteps:
            noised_image, _ = self.noise_images(image, torch.LongTensor([t]))
            denoised_image = self.sample(model, noised_image.shape[0], start_x=noised_image, start_t=t)
            denoised_images.append(denoised_image)
            noised_images.append(noised_image)
        return torch.cat(noised_images + denoised_images)


def train(args):
    device = args.device
    dataloader = get_data(args)
    model = UNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)  # change upper noise level
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        static_images = next(iter(dataloader))[0]  # overfit single batch
        for i, (images, _) in enumerate(pbar):
            # images = images.to(device)
            images = static_images.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch*l + i)

        # images = next(iter(dataloader))[0].to(device)
        sampled_images = diffusion.sample(model, n=images.shape[0])
        save_image(sampled_images.add(1).mul(0.5), os.path.join("results", f"{epoch}.jpg"))
        t = torch.Tensor([50, 100, 150, 200, 300, 600, 700, 999]).long().to(device)
        denoised_images = diffusion.sample_from_timestep(model, images[0].unsqueeze(0), t)
        save_image(denoised_images.add(1).mul(0.5), os.path.join("results", f"{epoch}_denoised.jpg"), nrow=len(t))


def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_test_overfit_denoising"
    args.epochs = 300
    args.batch_size = 16  # 5
    args.image_size = 64
    args.dataset_path = r"C:\Users\dome\datasets\landscape_img_folder"
    args.device = "cuda"
    args.lr = 3e-4
    train(args)


if __name__ == '__main__':
    # n = Noise()
    # img = torch.Tensor(np.array(Image.open("./images/test.jpg").resize((256, 256)))).permute(2, 0, 1) / 127.5 - 1
    #
    # imgs = img.unsqueeze(0).expand(3, -1, -1, -1)
    # ts = torch.Tensor([100, 400, 300]).long()
    # noised_imgs = n.noise_images(imgs, ts)
    # plt.imshow(noised_imgs[0].add(1).mul(0.5).permute(1, 2, 0))
    # plt.show()
    launch()
