from utils import get_alphabet
from ddpm_conditional import *

config.epochs = 100
config.noise_steps = 1000
config.num_classes = 26
config.img_size = 32
config.batch_size = 320
config.slice_size = 1
config.log_every_epoch = 2
config.do_validation = False

train_dl, _ = get_alphabet(config)

                
adiff = Diffusion(config.noise_steps, img_size=config.img_size, num_classes=config.num_classes, c_in=1, c_out=1)

with wandb.init(project="train_fonts", config=config):
    adiff.fit(config)