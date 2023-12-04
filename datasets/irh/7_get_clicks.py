import cv2
import numpy as np
from os import listdir
from os.path import isfile, join


cv2.namedWindow("NIR")
cv2.namedWindow("RGB")
cv2.namedWindow("MAT")#, cv2.WINDOW_AUTOSIZE)
save = False
cursor  = (0, 0)
dim = (1280, 720)
img_path = "files/img_raw/"
click = 5
labels_path = "dataset.csv"


start = 764
material_id = 11
patch_half_width = 150

dict = {0:"brick", 1:"Ceramic", 2:"Fabric", 3:"Glass", 4:"Leather", 5:"Metal",
		6:"Mirror", 7:"Other",  8:"nothing", 9:"Paper", 10:"Plastic", 11:"Stone",
		12:"Tile", 13:"Wood"}

def mousecb(event, x, y, flags, userdata):
	global point
	global save
	global cursor
	if event == 4:
		save = True
	if x < patch_half_width:
		x = patch_half_width
	elif x > dim[0]-patch_half_width:
		x = dim[0]-patch_half_width
	if y < patch_half_width:
		y = patch_half_width
	elif y > dim[1]-patch_half_width:
		y = dim[1]-patch_half_width
	cursor = (x, y)


def func_mat(id):
	global material_id
	material_id = id
	name = dict[id]
	cv2.setWindowTitle("MAT", "MAT " + name)

if start == -1:
	start = 0
	output_file = open(labels_path, "w")
	output_file.write("img_id,x,y,mat\n")

cv2.createTrackbar('Material', 'MAT', material_id, 13, func_mat)
cv2.setMouseCallback("RGB", mousecb)
l = listdir(img_path + "rgb")
l.sort()

i = start
while i < len(l):
	n_rgb = l[i]
	img_id = n_rgb.split('_')[0]
	print(i, ') img', img_id)
	n_dpt = img_path + 'dpt/' + img_id + '_dpt.png'
	n_nir = img_path + 'nir/' + img_id + '_nir.png'
	n_rgb = img_path + 'rgb/' + n_rgb

	img_dpt = cv2.imread(n_dpt)
	img_nir = cv2.imread(n_nir)
	img_rgb = cv2.imread(n_rgb)


	while True:
		output_file = open(labels_path, "a")
		img_nir_ = img_nir.copy()
		img_rgb_ = cv2.resize(img_rgb, (1280, 720))
		
		cv2.circle(img_rgb_, cursor, 1, (0,0,255), 1)
		cv2.circle(img_rgb_, cursor, 75, (0,255,0), 1)
		cv2.circle(img_rgb_, cursor, patch_half_width, (255,0,0), 1)

		#505, 672
		cv2.imshow("NIR", img_nir_[99:-116, 297:-311, :])
		cv2.imshow("RGB", img_rgb_)

		if save:
			#save in text format
			x = cursor[0]/dim[0]
			y = cursor[1]/dim[1]
			output = img_id + ',' + str(x) + ',' + str(y) + ',' + str(material_id) + '\n'
			output_file.write(output)
			click = click + 1
			save = False
			print(click, "clicks")
		k = cv2.waitKey(1)
		if k == ord(" "):
			break
		elif k == 27:
			output_file.close()
			exit(0)
		elif k == 225:
			i -= 2
			break
	i += 1
