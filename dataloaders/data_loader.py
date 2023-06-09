from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.io import ImageReadMode
import torchvision.transforms as T
import os
import cv2
import torch
import numpy as np

class VAEDataset(Dataset):
    def __init__(self, rgb_dir, ir__dir, half=True):
        self.HALF = half
        self.rgb_dir = rgb_dir
        self.rgb_list = os.listdir(rgb_dir)
        self.rgb_list.sort()

        self.ir__dir = ir__dir
        self.ir__list = os.listdir(ir__dir)
        self.ir__list.sort()
        self.rgb_ycrcb_M = torch.tensor([[0.299, 0.587, 0.114], [-0.169, -0.331, 0.500], [0.5, -0.419, -0.081]]).T.reshape(1,3,3)
        self.rgb_ycrcb_D = torch.tensor([0.0, 0.5, 0.5])

    def __len__(self):
        return len(self.rgb_list)

    def __getitem__(self, index):
        rgb_file = self.rgb_list[index]
        file_id = int(rgb_file.split('.')[0].split('_')[1])
        ir__file = "FLIR_{:05}.jpeg".format(file_id)

        ir__img = read_image(self.ir__dir + ir__file, ImageReadMode.GRAY)/255
        ir__img = T.Resize([256,256])(ir__img)

        rgb_img = read_image(self.rgb_dir + rgb_file, ImageReadMode.RGB)/255
        rgb_img = rgb_img[:, 70:-70, 70:-70]
        rgb_img = (rgb_img.permute(1,2,0) @ self.rgb_ycrcb_M + self.rgb_ycrcb_D).permute(2,0,1)
        rgb_img = T.Resize([256,256])(rgb_img)
        if self.HALF:
            return ir__img.half(), rgb_img.half()
        else:
            return ir__img, rgb_img


class VAEDataset1(Dataset):
    def __init__(self, rgb_dir, ir__dir, half=False):
        self.HALF = half
        self.rgb_dir = rgb_dir
        self.rgb_list = os.listdir(rgb_dir)
        self.rgb_list.sort()

        self.ir__dir = ir__dir
        self.ir__list = os.listdir(ir__dir)
        self.ir__list.sort()
        self.rgb_ycrcb_M = torch.tensor([[0.299, 0.587, 0.114], [-0.169, -0.331, 0.500], [0.5, -0.419, -0.081]]).T.reshape(1,3,3)
        self.rgb_ycrcb_D = torch.tensor([0.0, 0.5, 0.5])

    def __len__(self):
        return len(self.rgb_list)

    def __getitem__(self, index):
        rgb_file = self.rgb_list[index]
        file_id = int(rgb_file.split('_')[0])
        ir__file = "{:06}_nir.png".format(file_id)

        ir__img = read_image(self.ir__dir + ir__file, ImageReadMode.GRAY)/255
        ir__img = T.Resize([256,256])(ir__img)

        rgb_img = read_image(self.rgb_dir + rgb_file, ImageReadMode.RGB)/255
        rgb_img = rgb_img[:, 70:-70, 70:-70]
        #rgb_img = (rgb_img.permute(1,2,0) @ self.rgb_ycrcb_M + self.rgb_ycrcb_D).permute(2,0,1)
        rgb_img = T.Resize([256,256])(rgb_img)
        if self.HALF:
            return ir__img.half(), rgb_img.half()
        else:
            return ir__img, rgb_img
