import torch
import pandas as pd
from os import walk, listdir
from torch.utils.data import Dataset
from torchvision.io import read_image
import cv2
import numpy as np

#Dataloader for SUN dataset providing depth and rgb unlabeled data.
class SUNDataset(Dataset):
    def __init__(self, dir):
        self.dir = dir if dir[-1] == '/' else dir + '/'
        self.data = []
        for p in [x[0][:-5]  for x in walk(self.dir ) if x[0].split('/')[-1] == 'image']:
            im_p = p + 'image/'
            im_p = im_p + listdir(im_p)[0]
            
            def_p = p + 'depth/'
            def_p = def_p + listdir(def_p)[0]
            
            deg_p = p + 'depth_bfx/'
            deg_p = deg_p + listdir(deg_p)[0]
            self.data.append((im_p, def_p, deg_p))
    

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):


        background = cv2.imread(self.data[index][1], -1)
        foreground = cv2.imread(self.data[index][2], -1)

        
        X_dpt = torch.Tensor((background + foreground)/65535)
        X_rgb = read_image(self.data[index][0])/255

        return X_rgb, X_dpt

#Dataloader for EPFL dataset providing nir and rgb unlabeled data.
class EPFLataset(Dataset):
    def __init__(self, dir):
        self.dir = dir if dir[-1] == '/' else dir + '/'
        self.data = []
        for d in [x[0] for x in walk(dir)][1:]:
            for f in listdir(d):
                if f[-8:-5] == 'rgb':
                    self.data.append(d + '/' + f[:-8])
    

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X_rgb = torch.Tensor(cv2.imread(self.data[index] + 'rgb.tiff'))/255
        X_nir = torch.Tensor(cv2.imread(self.data[index] + 'nir.tiff'))/255

        return X_rgb, X_nir
