import glob
from models.googlenet import googlenet
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import sys
import torch
import torch.nn as nn
from models.coatnet2_multimodal import coatnet_full

import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils

from torchvision.models import resnet50

color_palette = np.loadtxt('./palette.txt').astype(np.uint8)
path = "../../datasets/irh/files/img_raw/rgb/"
ch = 23
minc = False
l = os.listdir(path)
sorted(l)

def color_image_w_masks(image, masks):
    image = image.astype(np.uint8)

    for index in range(ch):
        mask = (masks == index).astype(np.uint8)
        if mask.sum() == 0:
            continue
            print("0")
        color = color_palette[index]
        mask = np.expand_dims(mask, axis=-1)
        mask = np.repeat(mask, 3, axis=-1)
        cv2.imwrite("masks/cat_" + str(index) + ".png", cv2.resize(mask, (640, 480))*150)
        mask = mask * np.array(color).reshape((-1, 3)) + (1 - mask) * image
        mask = mask.astype(np.uint8)
        image = cv2.addWeighted(image, .5, mask, .5, 0)
    return image

def inference_on_whole_image(img, model):
    h,w,c = img.shape
    if h % 256 != 0:
        h_ = (h // 256 + 1) * 256
    else:
        h_ = h
    if w %  256 != 0:
        w_ = (w // 256 + 1) * 256
    else:
        w_ = w
    nh = h_ // 128
    nw = w_ // 128

    #load tensor and resize
    img = cv2.resize(img, (w_, h_))
    img = img.astype(np.float32).transpose(2,0,1)
    if minc:
        img[0,:,:] -= 104
        img[1,:,:] -= 117
        img[2,:,:] -= 124
        img = torch.FloatTensor(img).unsqueeze(0).cuda()
    else:
        img = torch.FloatTensor(img/255).unsqueeze(0).cuda()
    softmax = nn.Softmax(dim=1)


    prob = np.zeros((h_, w_, ch))

    for i in range(nh-1):
        for j in range(nw-1):
            img_patch = img[:,:,i*128:(i+2)*128, j*128:(j+2)*128]
            #add imshow rectangle on the image
            pred = model(img_patch)
            pred = softmax(pred)
            if minc:
                pred = pred.squeeze()
                pred = pred.cpu().numpy().transpose(1,2,0)
            else:
                pred = pred.unsqueeze(0)
                pred = pred.cpu().numpy()
            #print(np.argmax(pred, axis=2))
            pred = cv2.resize(pred, (256, 256))
            prob[i*128:(i+2)*128, j*128:(j+2)*128, :] = pred

    return prob

def multi_scale_inference(img, model):
    h,w,c = img.shape
    scales = [.5, 1, 1.5]
    prob = np.zeros((h,w,ch))

    for scale in scales:
        img_ = cv2.resize(img, (int(w*scale), int(h*scale)))
        prob_ = inference_on_whole_image(img_, model)
        prob += cv2.resize(prob_, (w,h))

    prob /= 3
    return prob

class DenseCRF(object):
    def __init__(self, iter_max, pos_w, pos_xy_std, bi_w, bi_xy_std, bi_rgb_std):
        self.iter_max = iter_max
        self.pos_w = pos_w
        self.pos_xy_std = pos_xy_std
        self.bi_w = bi_w
        self.bi_xy_std = bi_xy_std
        self.bi_rgb_std = bi_rgb_std

    def __call__(self, image, probmap):
        C, H, W = probmap.shape

        U = utils.unary_from_softmax(probmap)
        U = np.ascontiguousarray(U)

        image = np.ascontiguousarray(image)

        d = dcrf.DenseCRF2D(W, H, C)
        d.setUnaryEnergy(U)
        d.addPairwiseGaussian(sxy=self.pos_xy_std, compat=self.pos_w)
        d.addPairwiseBilateral(
            sxy=self.bi_xy_std, srgb=self.bi_rgb_std, rgbim=image, compat=self.bi_w
        )

        Q = d.inference(self.iter_max)
        Q = np.array(Q).reshape((C, H, W))

        return Q


print()
#2,6,10, 33
i = int(sys.argv[1])
ch = 15

m = coatnet_full(0)
m.load_state_dict(torch.load('../../weights/coatnet2_rgb_irh.pth'), strict=False)
"""
m = resnet50()
m.fc = nn.Linear(m.fc.in_features, ch)
m.load_state_dict(torch.load('../../weights/resnet50_minc.pth'), strict=False)
m.cuda().eval()

m = googlenet()
m.load_state_dict(torch.load('../../weights/minc-googlenet.pth'), strict=False)
m.cuda().eval()
"""
torch.set_grad_enabled(False)
if ch ==23:
    labels = open('categories.txt', 'r').readlines()
else:

    labels = open('categories_irh.txt', 'r').readlines()
labels = [i.strip() for i in labels]

postprocessor = DenseCRF(
    iter_max=10,
    pos_xy_std=1,
    pos_w=3,
    bi_xy_std=67,
    bi_rgb_std=3,
    bi_w=4,
)
print(l[i])
img = cv2.imread(path + l[i])
img = cv2.resize(img, (1280, 720))

prob0 = multi_scale_inference(img, m)
#prob1 = multi_scale_inference(img, m1)
prob = prob0 #(prob0 + prob1) / 2

prob = cv2.resize(prob, (480, 320))
img = cv2.resize(img, (480, 320))
prob = prob.transpose(2,0,1)

prob = postprocessor(img, prob)
labelmap = np.argmax(prob, axis=0)



mask = color_image_w_masks(img, labelmap)

cv2.imwrite("minc_inf.png", cv2.resize(mask, (1280, 720)))

img = np.concatenate([img, mask], axis=1)
#cv2.imshow('img', img)
#cv2.waitKey()

#'''
plt.figure(figsize=(10, 10))
plt.imshow(img[:, :, ::-1])

plt.figure(figsize=(15, 15))

for i in range(ch):
    mask = labelmap == i
    ax = plt.subplot(4, 6, i + 1)
    ax.set_title(labels[i])
    ax.imshow(img[:, :, ::-1])
    ax.imshow(mask.astype(np.float32), alpha=0.5)
    ax.axis("off")

plt.tight_layout()
plt.show()
