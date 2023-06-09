import torch
import torch.nn as nn

class Fusion(nn.Module):
    def __init__(self, device='cuda'):
        super(Fusion, self).__init__()
        self.device = torch.device(device)
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to(device) # hack to get sampling on the GPU
        self.N.scale = self.N.scale.to(device)

    def forward(self, xr, xv):
        mu_v = xv.mean(dim=(2,3))
        sigma_v = xv.std(dim=(2,3))

        mu_r = xr.mean(dim=(2,3))
        sigma_r = xr.std(dim=(2,3))

        mu = (mu_r * torch.square(sigma_v) + mu_v * torch.square(sigma_r)) / (torch.square(sigma_v) + torch.square(sigma_r))
        sigma = (sigma_r * sigma_v) / torch.sqrt(torch.square(sigma_v) + torch.square(sigma_r))

        eps = self.N.sample(mu.shape)
        z = mu + sigma * eps
        del eps, sigma_r, mu_r, sigma_v, mu_v, xr, xv
        return z, mu, sigma


class Encoder(nn.Module):
    def __init__(self, in_layers=1, device='cuda'):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(in_layers, 64, 5, stride=2, padding=2)
        self.batch1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, 5, stride=2, padding=2)
        self.batch2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, 5, stride=2, padding=2)
        self.batch3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256, 256, 5, stride=2, padding=2)

        self.ReLu = torch.nn.LeakyReLU()

    def forward(self, x):
        x1 = self.ReLu(self.batch1(self.conv1(x)))
        x2 = self.ReLu(self.batch2(self.conv2(x1)))
        x  = self.ReLu(self.batch3(self.conv3(x2)))
        x  = self.ReLu(self.conv4(x))

        return (x1, x2, x)


class Decoder(nn.Module):
    def __init__(self, device='cuda', full=True):
        super(Decoder, self).__init__()

        self.upsample = nn.Upsample(scale_factor=32, mode='nearest')

        self.deconv1_2 = nn.ConvTranspose2d(256, 256, 4, 2, 1)
        self.batch1_2 = nn.BatchNorm2d(256)

        self.deconv2 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.batch2 = nn.BatchNorm2d(128)

        self.deconv3 = nn.ConvTranspose2d(128, 64, 1, 1)
        self.batch3 = nn.BatchNorm2d(64)

        self.deconv4 = nn.ConvTranspose2d(64, 3, 2, 2)

        self.ReLu = torch.nn.GELU()

        self.FULL = full

    def forward(self, z, xr1, xr2, xv1, xv2):

        z = self.upsample(z.view(-1,256,1,1))

        z = self.ReLu(self.batch1_2(self.deconv1_2(z)))

        z = z + torch.concat((xv2, xr2), dim=-3)
        if not self.FULL:
            return z
        z = self.deconv2(z)
        z = self.ReLu(self.batch2(z))

        z = self.deconv3(z + torch.concat((xv1, xr1), dim=-3))
        z = self.ReLu(self.batch3(z))

        z = self.deconv4(z)
        z = torch.nn.Sigmoid()(z)
        return z

class Variational_Autoencoder(nn.Module):
    def __init__(self, full=True):
        super(Variational_Autoencoder, self).__init__()

        self.encoder_v = Encoder(in_layers=3)
        self.encoder_r = Encoder()
        self.fusion = Fusion()
        self.decoder = Decoder(full=full)

        self.fusion.trainble = False

    def forward(self, Ir, Iv):
        xr1, xr2, xr3 = self.encoder_r(Ir)
        xv1, xv2, xv3 = self.encoder_v(Iv)


        z, mu, sigma = self.fusion(xr3, xv3)
        z = self.decoder(z, xr1, xr2, xv1, xv2)

        del xr3, xv3
        return z, mu, sigma

class Variational_Autoencoder_bb(nn.Module):
    def __init__(self):
        super(Variational_Autoencoder_bb, self).__init__()

        self.adapt = nn.Sequential(nn.Conv2d(256, 128, 3, 1, 1),
                                   nn.Conv2d(128, 64, 3, 1, 1))
        self.vae = Variational_Autoencoder(full = False)


    def forward(self, Ir, Iv):
        z, mu, sigma = self.vae(Ir, Iv)
        z = self.adapt(z)
        return z, mu, sigma
