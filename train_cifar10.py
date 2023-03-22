from types import SimpleNamespace

import wandb
from ddpm_conditional import Diffusion
from utils import get_cifar


# Trains a conditional diffusion model on CIFAR10
# This is a very simple example, for more advanced training, see `ddp_conditional.py`

config = SimpleNamespace(    
    run_name = "cifar10_ddpm_conditional",
    epochs = 25,
    noise_steps=1000,
    seed = 42,
    batch_size = 128,
    img_size = 32,
    num_classes = 10,
    dataset_path = get_cifar(img_size=32),
    train_folder = "train",
    val_folder = "test",
    device = "cuda",
    slice_size = 1,
    do_validation = True,
    fp16 = True,
    log_every_epoch = 10,
    num_workers=10,
    lr = 5e-3)

diff = Diffusion(noise_steps=config.noise_steps , img_size=config.img_size)

with wandb.init(project="train_sd", group="train", config=config):
    diff.prepare(config)
    diff.fit(config)