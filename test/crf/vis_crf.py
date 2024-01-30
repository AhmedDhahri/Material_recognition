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
    h,w,c = img[0].shape
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
    for i in range(len(img)):
        img[i] = cv2.resize(img[i], (w_, h_))
        img[i] = img[i].astype(np.float32).transpose(2,0,1)
        img[i] = torch.FloatTensor(img[i]/255).unsqueeze(0).cuda()


    softmax = nn.Softmax(dim=1)
    prob = np.zeros((h_, w_, ch))

    if EXPERIEMT == 0:
        rgb = img[0]
    elif EXPERIEMT == 1:
        rgb, nir = img
    elif EXPERIEMT == 2:
        rgb, nir, dpt = img

    for i in range(nh-1):
        for j in range(nw-1):
            rgb_patch = rgb[:,:,i*size//2:(i+2)*size//2, j*size//2:(j+2)*size//2]
            nir_patch = nir[:,:,i*size//2:(i+2)*size//2, j*size//2:(j+2)*size//2] if EXPERIEMT > 0 else torch.FloatTensor(0)
            dpt_patch = dpt[:,:,i*size//2:(i+2)*size//2, j*size//2:(j+2)*size//2] if EXPERIEMT > 1 else torch.FloatTensor(0)
            #add imshow rectangle on the image
            pred = model(rgb_patch, nir_patch, dpt_patch)
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
    h,w,_ = img[0].shape
    scales = [.5, 1, 1.5]
    prob = np.zeros((h,w,ch))

    for scale in scales:

        img_ = []
        for i in range(len(img)):
            img_.append(cv2.resize(img[i], (int(w*scale), int(h*scale))))
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

def get_path():
    path, l = [], sys.path[0].split('/')
    for i in range(len(l)):
        path.append(l[i])
        if l[i] == 'Material_recognition':
            break
    return'/'.join(path) + '/'

img_id, EXPERIEMT, ch, size = int(sys.argv[2]), int(sys.argv[1]), 15, 384
path = get_path()
path_rgb = path + "datasets/irh/files/img_raw/rgb/"
color_palette = np.loadtxt(path + "test/crf/palette.txt").astype(np.uint8)
l = os.listdir(path_rgb)
sorted(l)

#2,6,10, 33
m = coatnet_full(EXPERIEMT, load=False)
if EXPERIEMT == 0:
    irh_checkpoint = "weights/coatnet2_rgb_irh.pth"
elif EXPERIEMT == 1:
    irh_checkpoint = "weights/coatnet2_rgb_nir_irh.pth"
elif EXPERIEMT == 2:
    irh_checkpoint = "weights/coatnet2_full_irh.pth"
m.load_state_dict(torch.load(path + irh_checkpoint), strict=False)

m = m.eval()
m = m.cuda()
labels = [i.strip() for i in open(path + "test/crf/categories_irh.txt", 'r').readlines()]

postprocessor = DenseCRF(
    iter_max=10,
    pos_xy_std=1,
    pos_w=3,
    bi_xy_std=67,
    bi_rgb_std=3,
    bi_w=4,
)

id = l[img_id].split('_')[0]
rgb = cv2.resize(cv2.imread(path_rgb + l[img_id]), (1280, 720))
nir = cv2.resize(cv2.imread(path_rgb + "../nir/{}_nir.png".format(id)), (1280, 720))
dpt = cv2.resize(cv2.imread(path_rgb + "../dpt/{}_dpt.png".format(id)) , (1280, 720))
img = [rgb, nir, dpt]

prob = cv2.resize(multi_scale_inference(img, m), (640, 360)).transpose(2,0,1)



img = cv2.resize(img[0], (640, 360))
prob = postprocessor(img, prob)
labelmap = np.argmax(prob, axis=0)

mask = color_image_w_masks(img, labelmap)
img = np.concatenate([img, mask], axis=1)

cv2.imwrite("crf_" + l[img_id], cv2.resize(mask, (1280, 720)))

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
