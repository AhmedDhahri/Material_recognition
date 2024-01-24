import cv2
import numpy as np
from os import listdir

window_name = "Fusion"
a, c, i = 184, 200, 0
alpha = 50
params = []

def on_trackbar_a(val):
    global a
    a = val
 
def on_trackbar_c(val):
    global c 
    c = val

def on_trackbar_alpha(val):
    global alpha 
    alpha = val
 

cv2.namedWindow(window_name)
cv2.createTrackbar("a", window_name , a, 500, on_trackbar_a)
cv2.createTrackbar("c", window_name , c, 500, on_trackbar_c)
cv2.createTrackbar("alpha", window_name , alpha, 100, on_trackbar_alpha)
'''
def reduce_distortion(img, k1, k2):
    distCoeff = np.zeros((4,1),np.float64)
    distCoeff[0,0] = (k1-500) * 1e-4
    distCoeff[1,0] = (k2-500) * -1e-4
    distCoeff[2,0] = 0.0
    distCoeff[3,0] = 0.0

    cam = np.eye(3,dtype=np.float32)
    cam[0,2] = img.shape[1]/2.0 # define center x
    cam[1,2] = img.shape[0]/2.0 # define center y
    cam[0,0] = 400. # define focal length x
    cam[1,1] = 400. # define focal length y

    return cv2.undistort(img, cam, distCoeff)

'''
img_list = listdir("files/img_raw/rgb")
while i < len(img_list):
    id = img_list[i].split('_')[0]
    path_rgb = "files/img_raw/rgb/" + id + "_rgb.png"
    path_nir = "files/img_raw/nir/" + id + "_nir.png"
    path_dpt = "files/img_raw/dpt/" + id + "_dpt.png"
    img_rgb = cv2.imread(path_rgb)
    if img_rgb.shape == (720, 1280, 3):
        a, c = 184, 200
    elif img_rgb.shape == (480, 640, 3):
        a,c = 295, 311
    img_rgb = cv2.resize(img_rgb, (1280, 720))
    img_nir = cv2.resize(cv2.imread(path_nir), (1280, 720))
    img_dpt = cv2.resize(cv2.imread(path_dpt), (1280, 720))

    corrected_nir = cv2.resize(img_nir[102:-112, a:-c-1, :], (1280, 720))
    fusion = np.uint8(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY) * (alpha/100.0) + cv2.cvtColor(corrected_nir, cv2.COLOR_RGB2GRAY) * (1.0 - alpha/100))
    cv2.imshow(window_name, fusion)
    key = cv2.waitKey(27)
    #enter13 space32
    if key == 27:#press escape
        break
    elif key == 13:#press enter
        params.append((id, a, 102, c, 113))
        i += 1
        print("{} - {}".format(i, id))
    elif key == 32 or key == 115:#press space
        i += 1
    elif key == 97:
        i-= 1
    elif key != -1:
        print(key, "key is invalid.\n", "Esc: quit, space: next, Enter: save params.")
print(params)


