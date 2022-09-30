from utils import get_alphabet
from ddpm_conditional import *

config.epochs = 100
config.noise_steps = 1000
config.num_classes = 26
config.img_size = 32
config.batch_size = 320
config.slice_size = 1
config.log_every_epoch = 2

train_dl, _ = get_alphabet(config)

class AlphabetDiffusion(Diffusion):
    def fit(self, args):
        setup_logging(args.run_name)
        device = args.device
        self.train_dataloader, _ = get_alphabet(args)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=0.001)
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=args.lr, 
                                                 steps_per_epoch=len(self.train_dataloader), epochs=args.epochs)
        self.mse = nn.MSELoss()
        self.ema = EMA(0.995)
        self.scaler = torch.cuda.amp.GradScaler()

        for epoch in range(args.epochs):
            logging.info(f"Starting epoch {epoch}:")
            _  = self.one_epoch(train=True, use_wandb=args.use_wandb)
            
            # log predicitons
            if epoch % args.log_every_epoch == 0:
                self.log_images(run_name=args.run_name, epoch=epoch, use_wandb=args.use_wandb)
                
adiff = AlphabetDiffusion(config.noise_steps, img_size=config.img_size, num_classes=config.num_classes, c_in=1, c_out=1)

with wandb.init(project="train_fonts", config=config):
    adiff.fit(config)