from torch.utils.data import Dataset
from torchvision.io import read_image
from torch.utils.data import DataLoader
import torch
import numpy as np
import cv2

class IRHDataset(Dataset):
    def __init__(self, dir, labels, size):
        self.dir = dir
        self.size = size
        self.data = np.genfromtxt(labels, delimiter=',')[1:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        Y = int(self.data[index][3])
        file_id = int(self.data[index][0])
        mod = "rgb"
        rgb_path = self.dir + '/' + "rgb" + '/' + "%06d_" % file_id + "rgb" + ".png"
        rgb_path = self.dir + '/' + "nir" + '/' + "%06d_" % file_id + "nir" + ".png"
        rgb_path = self.dir + '/' + "dpt" + '/' + "%06d_" % file_id + "dpt" + ".png"
        rgb = cv2.imread(rgb_path)


        h, w, _ = rgb.shape

        x = float(self.data[index][1])
        x = int(x * w)
        y = float(self.data[index][2])
        y = int(y * h)
        s = 25 + self.size[0]//2
        if x < s:
            s = x
        elif x > w-s:
            s = w-x
        if y < s:
            s = y
        elif y > h-s:
            s = h-y

        X = rgb[y-s:y+s, x-s:x+s, :]/255.0


        X = np.float32(cv2.resize(X, self.size))
        X = torch.from_numpy(X).permute(2, 0, 1)
        return X, Y

dataset = IRHDataset("../datasets/irh/files/img_raw", "../datasets/irh/dataset.csv", (256, 256))
loader = DataLoader(dataset=dataset, batch_size=4, num_workers=0, pin_memory=False, shuffle=True)
X, Y = next(iter(loader))
print(X.shape,Y)
