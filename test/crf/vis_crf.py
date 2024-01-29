import sys
sys.path.append(sys.path[0] + "/../..")
import glob
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils

from models.googlenet import googlenet
from torchvision.models import resnet50
from models.coatnet2_multimodal import coatnet_full
torch.set_grad_enabled(False)


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
    if h % size != 0:
        h_ = (h // size + 1) * size
    else:
        h_ = h
    if w %  size != 0:
        w_ = (w // size + 1) * size
    else:
        w_ = w
    nh = h_ // (size//2)
    nw = w_ // (size//2)

    #load tensor and resize
    img = cv2.resize(img, (w_, h_))
    img = img.astype(np.float32).transpose(2,0,1)
    if ch == 23:
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
            img_patch = img[:,:,i*size//2:(i+2)*size//2, j*size//2:(j+2)*size//2]
            #add imshow rectangle on the image
            xnir, xdpt = torch.Tensor(0).cuda(), torch.Tensor(0).cuda()
            pred = model(img_patch, xnir, xdpt)
            pred = softmax(pred)
            if ch == 23:
                pred = pred.squeeze()
                pred = pred.cpu().numpy().transpose(1,2,0)
            else:
                pred = pred.unsqueeze(0)
                pred = pred.cpu().numpy()
            #print(np.argmax(pred, axis=2))
            pred = cv2.resize(pred, (size, size))
            prob[i*size//2:(i+2)*size//2, j*size//2:(j+2)*size//2, :] = pred

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


path, l = [], sys.path[0].split('/')
for i in range(len(l)):
    path.append(l[i])
    if l[i] == 'Material_recognition':
        break
path = '/'.join(path) + '/'


path_img = path + "datasets/irh/files/img_raw/rgb/"
color_palette = np.loadtxt(path + "test/crf/palette.txt").astype(np.uint8)
ch, size = 15, 384
l = os.listdir(path_img)
sorted(l)

#2,6,10, 33
m = coatnet_full(0, load=False) if ch == 15 else googlenet()

if ch == 15:
    m.load_state_dict(torch.load(path + "weights/coatnet2_rgb_irh.pth"), strict=False)
else:
    m.load_state_dict(torch.load('../../weights/minc-googlenet.pth'), strict=False)
m = m.eval()
m = m.cuda()
"""
m = resnet50()
m.fc = nn.Linear(m.fc.in_features, ch)
m.load_state_dict(torch.load('../../weights/resnet50_minc.pth'), strict=False)
m.cuda().eval()

"""

if ch == 23:
    labels = open(path + "test/crf/categories.txt", 'r').readlines()
else:
    labels = open(path + "test/crf/categories_irh.txt", 'r').readlines()
labels = [i.strip() for i in labels]

postprocessor = DenseCRF(
    iter_max=10,
    pos_xy_std=1,
    pos_w=3,
    bi_xy_std=67,
    bi_rgb_std=3,
    bi_w=4,
)
print(path + l[int(sys.argv[1])])
img = cv2.imread(path_img + l[int(sys.argv[1])])
img = cv2.resize(img, (1280, 720))

prob = cv2.resize(multi_scale_inference(img, m), (640, 360)).transpose(2,0,1)
img = cv2.resize(img, (640, 360))

prob = postprocessor(img, prob)
labelmap = np.argmax(prob, axis=0)

mask = color_image_w_masks(img, labelmap)
img = np.concatenate([img, mask], axis=1)

cv2.imwrite("crf_" + l[int(sys.argv[1])], cv2.resize(mask, (1280, 720)))

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
