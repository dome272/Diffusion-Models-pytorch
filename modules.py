import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetEncoderBlock(nn.Module):
    def __init__(self, c_in, c_out, k=3, pad=1, stride=2, first_layer=False, use_incr=False):
        super().__init__()
        incr = 0 if (first_layer != True or use_incr == False) else 4
        self.encoder = nn.Sequential(
            nn.Identity() if first_layer else nn.ReLU(),
            nn.Conv2d(c_in, c_out, (k + 1 + incr), padding=(pad + incr // 2), stride=stride),
            nn.Identity() if first_layer else nn.InstanceNorm2d(c_out),
            nn.ReLU(),
            nn.Conv2d(c_out, c_out, k, padding=pad),
            nn.InstanceNorm2d(c_out)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x


class UNetDecoderBlock(nn.Module):
    def __init__(self, c_in, c_out, k=3, pad=1, stride=2, last_layer=False, use_incr=False):
        super().__init__()
        incr = 0 if (last_layer != True or use_incr == False) else 4
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(c_in, c_in, (k + 1), padding=pad, stride=stride),
            nn.InstanceNorm2d(c_in),
            nn.ReLU(),
            nn.Conv2d(c_in, c_out, (k + incr), padding=(pad + incr // 2)),
            nn.Identity() if last_layer else nn.InstanceNorm2d(c_out),
            nn.Identity() if last_layer else nn.ReLU()
        )

    def forward(self, x):
        x = self.decoder(x)
        return x


class UNet(nn.Module):
    def __init__(self, c_in=3, c_out=3, t_emb=128, c_t=128, hidden_size=256):
        super().__init__()
        self.c_t = c_t
        self.encoder_mappers = nn.ModuleList([
            nn.Conv2d(c_t, t_emb, 1) for _ in range(3)
        ])
        self.decoder_mappers = nn.ModuleList([
            nn.Conv2d(c_t, t_emb, 1) for _ in range(3)
        ])
        self.encoders = nn.ModuleList([
            UNetEncoderBlock(c_in, (hidden_size // 8), first_layer=True),
            UNetEncoderBlock(hidden_size // 8 + t_emb, hidden_size // 4),
            UNetEncoderBlock(hidden_size // 4 + t_emb, hidden_size // 2),
            UNetEncoderBlock(hidden_size // 2 + t_emb, hidden_size),
        ])
        self.decoders = nn.ModuleList([
            UNetDecoderBlock(hidden_size + t_emb, hidden_size // 2),
            UNetDecoderBlock(2 * hidden_size // 2 + t_emb, hidden_size // 4),
            UNetDecoderBlock(2 * hidden_size // 4 + t_emb, hidden_size // 8),
            UNetDecoderBlock((2 * hidden_size // 8), c_out, last_layer=True),
        ])

    def gen_t_embedding(self, t, max_positions=10000):
        half_dim = self.c_t // 2
        emb = math.log(max_positions) / (half_dim - 1)
        emb = torch.arange(half_dim, device=t.device).float().mul(-emb).exp()
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        if self.c_t % 2 == 1:  # zero pad
            emb = nn.functional.pad(emb, (0, 1), mode='constant')
        return emb[:, :, None, None]

    def forward(self, x, t):
        t = self.gen_t_embedding(t)  # bs x 128 x 1 x 1
        encodings = []
        for i, encoder in enumerate(self.encoders):
            if i > 0:
                c = self.encoder_mappers[i - 1](t).expand(-1, -1, *x.shape[2:])
                x = torch.cat([x, c], dim=1)
            x = encoder(x)
            encodings.insert(0, x)

        for i, decoder in enumerate(self.decoders):
            if i > 0:
                x = torch.cat((x, encodings[i]), dim=1)
            if i != len(self.decoders) - 1:
                c = self.decoder_mappers[i - 1](t).expand(-1, -1, *x.shape[2:])
                x = torch.cat([x, c], dim=1)
            x = decoder(x)

        return x


if __name__ == '__main__':
    net = UNet()
    x = torch.randn(1, 3, 256, 256)
    t = x.new_tensor([500] * x.shape[0]).long()
    print(net(x, t).shape)