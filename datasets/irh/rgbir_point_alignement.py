import cv2
import numpy as np
from os import listdir

window_name_rgb, window_name_nir = "RGB", "NIR"
data, pts_rgb, pts_nir = {}, [], []

def callback_rgb(event, x, y, flags, data):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("pts_rgb[{}] = {}".format(len(pts_rgb), (x, y)))
        pts_rgb.append((x, y))

def callback_nir(event, x, y, flags, data):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("pts_nir[{}] = {}".format(len(pts_nir), (x, y)))
        pts_nir.append((x, y))


cv2.namedWindow(window_name_rgb)
cv2.namedWindow(window_name_nir)

cv2.setMouseCallback(window_name_rgb, callback_rgb)
cv2.setMouseCallback(window_name_nir, callback_nir)

for f in listdir("img_raw/rgb"):
    id = f.split('_')[0]
    path_rgb = "img_raw/rgb/" + id + "_rgb.png"
    path_nir = "img_raw/nir/" + id + "_nir.png"
    path_dpt = "img_raw/dpt/" + id + "_dpt.png"
    img_rgb = cv2.resize(cv2.imread(path_rgb), (1280, 720))
    img_nir = cv2.resize(cv2.imread(path_nir), (1280, 720))
    img_dpt = cv2.resize(cv2.imread(path_dpt), (1280, 720))

    

    cv2.imshow(window_name_rgb, img_rgb)
    cv2.imshow(window_name_nir, img_nir)
    key = cv2.waitKey(0)
    if key == 27:#press escape
        break
    elif key == 13:#press enter
        data[id] = (pts_nir, pts_rgb)
        print(id)
    elif key == 32:#press space
        continue
    elif key != -1:
        print("Esc: quit, space: next, Enter: save params.")
print(data)


