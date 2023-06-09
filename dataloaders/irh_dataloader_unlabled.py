from torch.utils.data import Dataset
from torchvision.io import read_image
import torch
import pandas as pd

class IRHDataset(Dataset):
    def __init__(self, dir, labels):
        self.dir = dir
        self.data = pd.read_csv(labels)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        x_id = self.data["img_id"][index]
        X_rgb = read_image(self.dir + 'rgb/%07d_rgb.png'%x_id)/255
        X_nir = read_image(self.dir + 'nir/%07d_nir.png'%x_id)[0:1,:,:]/255
        X_dpt = read_image(self.dir + 'dpt/%07d_dpt.png'%x_id)/255

        Y = int(self.data["material_id"][index])

        return X_rgb, X_nir, X_dpt, Y
