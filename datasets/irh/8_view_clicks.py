import cv2
import numpy as np

material_id = 0
data = np.genfromtxt('dataset.csv', delimiter=',')[1:]
dict = {0:"brick", 1:"Ceramic", 2:"Fabric", 3:"Glass", 4:"Leather", 5:"Metal",
		6:"Mirror", 7:"Other",  8:"painted", 9:"Paper", 10:"Plastic", 11:"Stone",
		12:"Tile", 13:"Wood"}
		
cv2.namedWindow("MAT")#, cv2.WINDOW_AUTOSIZE)

def func_mat(id):
	global material_id
	material_id = id
	name = dict[id]
	cv2.setWindowTitle("MAT", "MAT " + name)
	
cv2.createTrackbar('Material', 'MAT', material_id, 13, func_mat)

i = 5400

while i < data.shape[0]:
	id = int(data[i][3])
	cls = dict[id]
	x, y = int(data[i][1] * 1280), int(data[i][2] * 720)
	
	img = cv2.imread('files/img_raw/rgb/%06d'%data[i][0] + '_rgb.png')
	img = cv2.resize(img, (1280, 720))
	
	cv2.circle(img, (x, y), 1, (0,0,255), 1)
	cv2.circle(img, (x, y), 75, (0,255,0), 1)
	cv2.circle(img, (x, y), 128, (255,0,0), 1)
	
	#print(i)#,  "rgb/%06d"%data[i][0] + "_rgb.png", data[i][1] , data[i][2], id )
	
	cv2.imshow("{} {}".format(cls, id), img)
	k = cv2.waitKey(0)
	if k == 13:
		print(i,  "rgb/%06d"%data[i][0] + "_rgb.png", data[i][1] , data[i][2], "from {} to {}".format(id, material_id))
	if k == ord(" "):
		print(i)
	if k == 27:
		break
	if k == 255:
		i = i-2
	i = i + 1
