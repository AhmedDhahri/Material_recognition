import torch
import torch.nn as nn


class Encoder(nn.Module):

    def __init__(self, in_layers=1):
        super().__init__()
        self.conv = nn.Conv2d(in_layers,16,3,1,1)

        self.d1 = nn.Conv2d(16,16,3,1,1)
        self.d2 = nn.Conv2d(32,16,3,1,1)
        self.d3 = nn.Conv2d(48,16,3,1,1)

        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.relu(self.conv(x))

        x1 = self.relu(self.d1(x))
        x = torch.concat([x, x1], dim=-3)


        x2 = self.relu(self.d2(x))
        x = torch.concat([x, x2], dim=-3)

        x3 = self.relu(self.d3(x))
        x = torch.concat([x, x3], dim=-3)

        del x1, x2, x3

        return x

class Fusion(nn.Module):

    def __init__(self):
        super().__init__()
        self.avg = nn.Conv2d(1,1,3,1,1)
        self.avg.weight = nn.Parameter(torch.ones(1,1,3,3)/9)


    def forward(self, x1, x2):
        c1 = self.avg(torch.norm(x1, p=1, dim=-3).reshape(-1,1,256,256))
        c2 = self.avg(torch.norm(x2, p=1, dim=-3).reshape(-1,1,256,256))
        x1 = (x1 * c1/ (c1 + c2)) + (x2 * c2/ (c1 + c2))

        del x2, c2, c1
        return x1

class Decoder(nn.Module):

    def __init__(self, out_layers=1):
        super().__init__()
        self.c2 = nn.Conv2d(64,64,3,1,1)
        self.c3 = nn.Conv2d(64,32,3,1,1)
        self.c4 = nn.Conv2d(32,16,3,1,1)
        self.c5 = nn.Conv2d(16,out_layers,3,1,1)

        self.relu = nn.ReLU()


    def forward(self, x):

        x = self.relu(self.c2(x))
        x = self.relu(self.c3(x))
        x = self.relu(self.c4(x))
        x = self.relu(self.c5(x))

        return x


class DenseFuse(nn.Module):

    def __init__(self, out_layers, full=True):
        super().__init__()

        self.encoder1 = Encoder(1)
        self.encoder2 = Encoder(3)
        self.fusion = Fusion()
        self.decoder = Decoder(out_layers)
        self.FULL = full

    def forward(self, x1, x2):

        x1 = self.encoder1(x1)
        x2 = self.encoder2(x2)

        x1 = self.fusion(x1, x2)
        del x2

        if self.FULL:
            x1 = self.decoder(x1)

        return x1




class DenseFuse_bb(nn.Module):

    def __init__(self, out_layers, full=True):
        super().__init__()

        self.df = DenseFuse(out_layers, full=False)
        self.adapt = nn.Sequential(nn.Conv2d(64, 64, 3, 2, 1),
                                  nn.Conv2d(64, 64, 3, 2, 1))

    def forward(self, x1, x2):

        x1 = self.df(x1, x2)
        del x2
        x1 = self.adapt(x1)

        return x1
