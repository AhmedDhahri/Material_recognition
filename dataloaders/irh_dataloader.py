from torch.utils.data import Dataset
from torchvision.io import read_image
from torch.utils.data import DataLoader
import torch
import numpy as np
import cv2
import random
from torchvision import transforms as v2

class IRHDataset(Dataset):
    def __init__(self, dir, labels, size, experiment=0):
        self.dir = dir
        self.size = size
        self.data = np.genfromtxt(labels, delimiter=',')[1:]
        self.experiment = experiment

        self.transforms = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.2),
        ])
        self.blur_transform = v2.RandomApply([v2.GaussianBlur(5)], p=0.5)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        Y, file_id = int(self.data[index][3]), int(self.data[index][0])
        rgb_path = self.dir + '/' + "rgb" + '/' + "%06d_" % file_id + "rgb" + ".png"
        nir_path = self.dir + '/' + "nir" + '/' + "%06d_" % file_id + "nir" + ".png"
        dpt_path = self.dir + '/' + "dpt" + '/' + "%06d_" % file_id + "dpt" + ".png"
        
        rgb = cv2.imread(rgb_path)
        (a, c, std, clip) = (185, 200, 6.5, 12) if rgb.shape == (1280, 720, 3) else (295, 315, 6.5, 9)
        a, c = int(max(min(random.normalvariate(mu=a, sigma=std), a+clip), a-clip)), int(max(min(random.normalvariate(mu=c, sigma=std), c+clip), c-clip))

        x, y = int(float(self.data[index][1]) * 1280), int(float(self.data[index][2]) * 720)
        resize, min_size = random.randrange(0, 50) + self.size[0]//2,  min(x,y, 1280-x, 720-y)
        if resize > min_size:
            resize = min_size
        
        if self.experiment == 0:
            if rgb.shape == (480, 640, 3):
                rgb = cv2.resize(rgb, (1280, 720))

            X_rgb = rgb[y-resize:y+resize, x-resize:x+resize, :]/255.0
            X_rgb = np.float32(cv2.resize(X_rgb, self.size))
            X_rgb = torch.from_numpy(X_rgb).permute(2, 0, 1)

            X_rgb = self.blur_transform(X_rgb)
            X_rgb = self.transforms(X_rgb)
            return X_rgb, Y

        if self.experiment == 1:
            nir = cv2.imread(nir_path)
            if rgb.shape == (480, 640, 3):
                rgb = cv2.resize(rgb, (1280, 720))
            nir = cv2.resize(nir[102:607, a:1280-c], (1280, 720))

            X_rgb = rgb[y-resize:y+resize, x-resize:x+resize]/255.0
            X_rgb = np.float32(cv2.resize(X_rgb, self.size))
            X_rgb = torch.from_numpy(X_rgb).permute(2, 0, 1)
            
            X_nir = nir[y-resize:y+resize, x-resize:x+resize]/255.0
            X_nir = np.float32(cv2.resize(X_nir, self.size))
            X_nir = torch.from_numpy(X_nir).permute(2, 0, 1)

            X_rgb = self.blur_transform(X_rgb)            
            state = torch.get_rng_state()
            X_rgb = self.transforms(X_rgb)
            torch.set_rng_state(state)
            X_nir = self.transforms(X_nir)
            return (X_rgb, X_nir), Y

        if self.experiment == 2:
            nir, dpt = cv2.imread(nir_path), cv2.imread(dpt_path)
            if rgb.shape == (480, 640, 3):
                rgb = cv2.resize(rgb, (1280, 720))
                dpt = cv2.resize(dpt, (1280, 720))
            nir = cv2.resize(nir[102:607, a:1280-c], (1280, 720))

            X_rgb = rgb[y-resize:y+resize, x-resize:x+resize]/255.0
            X_rgb = np.float32(cv2.resize(X_rgb, self.size))
            X_rgb = torch.from_numpy(X_rgb).permute(2, 0, 1)
            
            X_nir = nir[y-resize:y+resize, x-resize:x+resize]/255.0
            X_nir = np.float32(cv2.resize(X_nir, self.size))
            X_nir = torch.from_numpy(X_nir).permute(2, 0, 1)

            X_dpt = dpt[y-resize:y+resize, x-resize:x+resize]/255.0
            X_dpt = np.float32(cv2.resize(X_dpt, self.size))
            X_dpt = torch.from_numpy(X_dpt).permute(2, 0, 1)

            X_rgb = self.blur_transform(X_rgb)
            
            state = torch.get_rng_state()
            X_rgb = self.transforms(X_rgb)
            torch.set_rng_state(state)
            X_nir = self.transforms(X_nir)
            torch.set_rng_state(state)
            X_dpt = self.transforms(X_dpt)
            return (X_rgb, X_nir, X_dpt), Y