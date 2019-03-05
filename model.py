import torch.nn as nn
import torch.nn.functional as F



class Encoder(nn.Module):

    def __init__(self, z=16):
        super(Encoder, self).__init__()
        self.z = z
        self.encode = nn.Sequential(
            nn.Conv2d(1, 16, 4, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(16, 32, 4, 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 64, 4, 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, z * 2, 4, 1)
        )

    def forward(self, inp):
        inp = self.encode(inp)
        inp = inp.view(-1, 2, self.z)
        return inp


class Decoder(nn.Module):

    def __init__(self, z=16):
        super(Decoder, self).__init__()
        self.z = z
        self.decode = nn.Sequential(
            nn.ConvTranspose2d(1, 64, 4, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(32, 16, 4, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(16, 1, 4, 2),
            nn.Sigmoid()
        )

    def forward(self, inp):
        inp = inp.view(-1, 1, 4, 4)
        inp = self.decode(inp)
        inp = inp.view(-1, 1, 28, 28)
        return inp