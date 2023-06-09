import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import sys
sys.path.append(sys.path[0]+'/../')
from models.denseFuse import DenseFuse_bb
from models.vae import Variational_Autoencoder_bb
from models.opticFlow import OF_AE_bb



class fusion_resnet(nn.Module):
    def __init__(self, fusion_model=0):
        super().__init__()
        self.fusion_model = fusion_model
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.fc = nn.Linear(self.resnet.fc.in_features, 23)
        if fusion_model != 0:
            self.resnet = nn.Sequential(*list(self.resnet.children())[4:-1])
        if fusion_model == 1:
            self.model = DenseFuse_bb(3)
        if fusion_model == 2:
            self.model = Variational_Autoencoder_bb()
        if fusion_model == 3:
            self.model = OF_AE_bb()

    def forward(self, Ir, Iv):
        if self.fusion_model == 0:
            Iv = self.resnet(Iv)
        elif self.fusion_model == 2:
            Iv, mu, sigma = self.model(Ir, Iv)
            Iv = self.resnet(Iv)
            Iv = torch.squeeze(Iv)
            Iv = self.fc(Iv)
            return Iv, mu, sigma
        else:
            Iv = self.model(Ir, Iv)
            Iv = self.resnet(Iv)
            Iv = torch.squeeze(Iv)
            Iv = self.fc(Iv)
        return Iv
