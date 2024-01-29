from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.io import ImageReadMode
import torchvision.transforms as T
import os
import cv2
import torch
import numpy as np
import random
import math
from PIL import Image

class MINCDataset(Dataset):
    def __init__(self, dir, labels, size=(256, 256), f=0.16, drop_cls=[5, 6, 8, 17, 18, 21, 22], cls=-1):
        if cls in drop_cls:
            print("Class {} is droped.".format(cls))
        else:
            print("Class {} is kept.".format(cls))
        
        self.dir = dir
        self.data = np.genfromtxt(labels, delimiter=',')
        self.size = size
        self.f = f

        print("Dataset size is {}".format(self.data.shape[0]), end='. ')
        data_ = []
        for i in range(self.data.shape[0]):
            m, _, x, y = self.data[i]
            if int(m + 1) != cls and cls != -1:
                continue
            if int(m + 1) in drop_cls:
                continue
            if x < f or y < f or (1-x) < f or (1-y) < f:
                continue
            data_.append(self.data[i])
        self.data = np.array(data_)
        print("After dropping classes size is {}".format(self.data.shape[0]))

        #consider data equilibribium
        

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        #11,000105677,0.18545454545454546,0.7845454545454545

        Y = int(self.data[index, 0])
        file_id = int(self.data[index, 1])
        sub_folder = str(file_id % 10)
        file_path = self.dir + '/' + sub_folder + '/' + "%09d" % file_id + ".jpg"
        img_ = cv2.imread(file_path)


        min_shape_div2 = int(min(img_.shape[:2]) * self.f)
        c_x = int(self.data[index, 2] * img_.shape[1])
        c_y = int(self.data[index, 3] * img_.shape[0])#maybe swap 2 and 3

        X = img_[c_y - min_shape_div2  : c_y + min_shape_div2,
                 c_x - min_shape_div2  : c_x + min_shape_div2,
                 :]
        X = np.float32(cv2.resize(X, self.size)/255.0)
        X = torch.from_numpy(X).permute(2, 0, 1)
        return X, Y


class MINCDataLoader:
    def __init__(self, dir, labels, batch_size, size=256, f=0.16):
        self.dir = dir
        self.data = np.genfromtxt(labels, delimiter=',')
        self.size = size
        self.f = f
        self.labels_id = []
        self.batch_size = batch_size
        self.data = np.genfromtxt(labels, delimiter=',')[1:]
        for i in range(self.data.shape[0]):
            c, img, x, y = self.data[i]
            if x < f or y < f or (1-x) < f or (1-y) < f:
                pass
            else:
                self.labels_id.append(i)
        self.size_ = (round(1.28 * size), round(1.28 * size))
        self.size = (size, size)
        random.shuffle(self.labels_id)
        self.transform = T.Compose([
            #T.Normalize(mean=(124, 117, 104)),
            T.RandomHorizontalFlip(),
            T.Resize(self.size_, antialias=True),
            T.RandomResizedCrop(size=size, scale=(0.71, 1.41), ratio=(0.75, 1.33), antialias=True)
        ])

    def __len__(self):
        return len(self.labels_id)//self.batch_size

    def __getitem__(self, index):
        Xa, Ya = [], []
        for e in range(self.batch_size):
            i = self.labels_id[index*self.batch_size + e]
            file_id = int(self.data[i][1])
            file_path = self.dir + '/' + str(file_id % 10) + '/' + "%09d" % file_id + ".jpg"

            img_ = None
            try:
                img_ = read_image(file_path, ImageReadMode.RGB)
            except:
                continue
            min_shape_div2 = int(min(img_.shape[1:]) * self.f)
            c_x = int(self.data[i][2] * img_.shape[2])
            c_y = int(self.data[i][3] * img_.shape[1])
            X = img_[:,
                    c_y - min_shape_div2  : c_y + min_shape_div2,
                    c_x - min_shape_div2  : c_x + min_shape_div2]
            Xa.append(self.transform(X)/255)
            Ya.append(torch.IntTensor([int(self.data[i][0])]))
        X = torch.stack(Xa, 0)
        Y = torch.stack(Ya, 0)
        return X, Y


class MINCDataset_gnet(Dataset):
    def __init__(self, dir, labels, size=(256, 256), f = 0.16):
        self.dir = dir
        self.data = np.genfromtxt(labels, delimiter=',')
        self.size = size
        self.labels_id = []
        self.f = f
        for i in range(self.data.shape[0]):
            c, img, x, y = self.data[i]
            if x < f or y < f or (1-x) < f or (1-y) < f:
                pass
            else:
                self.labels_id.append(i)

        #consider data equilibribium
    def __len__(self):
        return len(self.labels_id)

    def __getitem__(self, i):
        #11,000105677,0.18545454545454546,0.7845454545454545
        index = self.labels_id[i]
        Y = int(self.data[index][0])
        file_id = int(self.data[index][1])
        sub_folder = str(file_id % 10)
        file_path = self.dir + '/' + sub_folder + '/' + "%09d" % file_id + ".jpg"
        img_ = cv2.imread(file_path)

        min_shape_div2 = int(min(img_.shape[:2]) * self. f)
        c_x = int(self.data[index][2] * img_.shape[1])
        c_y = int(self.data[index][3] * img_.shape[0])#maybe swap 2 and 3

        X = img_[c_y - min_shape_div2  : c_y + min_shape_div2,
                 c_x - min_shape_div2  : c_x + min_shape_div2,
                 :]
        X = np.float32(cv2.resize(X, self.size))
        X = torch.from_numpy(X).permute(2, 0, 1)
        X[0, :, :] = X[0, :, :] - 104
        X[1, :, :] = X[1, :, :] - 117
        X[2, :, :] = X[2, :, :] - 124
        return X, Y
