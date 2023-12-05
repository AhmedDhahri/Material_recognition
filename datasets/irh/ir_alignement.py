import cv2
import numpy as np
from os import listdir

window_name = "Fusion"
a, b, c, d, k1, k2, i = 188, 104, 198, 117, 500, 500, 0
alpha = 50
params = []

def on_trackbar_a(val):
    global a
    a = val
 
def on_trackbar_b(val):
    global b 
    b = val
 
def on_trackbar_c(val):
    global c 
    c = val
 
def on_trackbar_d(val):
    global d 
    d = val

def on_trackbar_k1(val):
    global k1
    k1 = val

def on_trackbar_k2(val):
    global k2 
    k2 = val

def on_trackbar_alpha(val):
    global alpha 
    alpha = val
 

cv2.namedWindow(window_name)
cv2.createTrackbar("a", window_name , a, 500, on_trackbar_a)
cv2.createTrackbar("b", window_name , b, 500, on_trackbar_b)
cv2.createTrackbar("c", window_name , c, 500, on_trackbar_c)
cv2.createTrackbar("d", window_name , d, 500, on_trackbar_d)
cv2.createTrackbar("k1", window_name , k1, 1000, on_trackbar_k1)
cv2.createTrackbar("k2", window_name , k2, 1000, on_trackbar_k2)
cv2.createTrackbar("alpha", window_name , alpha, 100, on_trackbar_alpha)

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


img_list = listdir("img_raw/rgb")
while i < len(img_list):
    id = img_list[i].split('_')[0]
    path_rgb = "img_raw/rgb/" + id + "_rgb.png"
    path_nir = "img_raw/nir/" + id + "_nir.png"
    path_dpt = "img_raw/dpt/" + id + "_dpt.png"
    img_rgb = cv2.resize(cv2.imread(path_rgb), (1280, 720))
    img_nir = cv2.resize(cv2.imread(path_nir), (1280, 720))
    img_dpt = cv2.resize(cv2.imread(path_dpt), (1280, 720))

    corrected_nir = reduce_distortion(img_nir, k1, k2)
    corrected_nir = cv2.resize(corrected_nir[b:-d-1, a:-c-1, :], (1280, 720))
    fusion = np.uint8(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY) * (alpha/100.0) + cv2.cvtColor(corrected_nir, cv2.COLOR_RGB2GRAY) * (1.0 - alpha/100))
    align = cv2.Canny(img_rgb, 100, 200) * 100 + np.uint8(cv2.cvtColor(corrected_nir, cv2.COLOR_RGB2GRAY) * (alpha / 100.0))
    #align = cv2.normalize(align, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imshow(window_name, fusion)
    key = cv2.waitKey(100)
    #enter13 space32
    if key == 27:#press escape
        break
    elif key == 13:#press enter
        params.append((id, a, b, c, d))
        i += 1
        print("{} - {}".format(i, id))
    elif key == 32:#press space
        i += 1
    elif key != -1:
        print("Esc: quit, space: next, Enter: save params.")
print(params)


