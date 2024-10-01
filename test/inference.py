import sys
sys.path.append(sys.path[0]+'/../')
sys.path.append(sys.path[0]+'/../utils')


import torch
import torch.nn as nn
from torchvision.io import read_image
from torchvision.transforms import Resize

from models.coatnet2_multimodal import coatnet_full
from model_params import model_params


#Arguments and init
MODEL_NAME, EXPERIMEMT, FILE,  = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
rgb, nir, dpt, SIZE, model, checkpoint = torch.Tensor(0).cuda(), torch.Tensor(0).cuda(), torch.Tensor(0).cuda(), None, None, None



#load model
if EXPERIMEMT == -1:
    model, checkpoint, _, SIZE, _ = model_params(model_name=MODEL_NAME, load=True).get() #"swinv2b", "vith14", "eva02l14", "maxvitxl", coatnet2
else:
    if EXPERIMEMT == 0:
        checkpoint = 'Material_recognition/weights/coatnet2_rgb_irh.pth' 
    elif EXPERIMEMT == 1:
        checkpoint = 'Material_recognition/weights/coatnet2_rgb_nir_irh.pth' 
    elif EXPERIMEMT == 2:
        checkpoint = 'Material_recognition/weights/coatnet2_full_irh.pth' 
    model, SIZE = coatnet_full(EXPERIMEMT, False), 384

model.load_state_dict(torch.load(checkpoint), strict=False)
model = model.eval()
model = model.cuda()

material_irh  = {0:'brick',1:'Ceramic',2:'Fabric',3:'Glass',4:'Leather',5:'Metal',6:'Mirror',7:'Other',8:'Painted',9:'Paper',10:'Plastic',11:'Stone',12:'Tile',13:'Wood',14:'Nothing'}
material_minc = {0:'brick',1:'carpet',2:'ceramic',3:'fabric',4:'foliage',5:'food',6:'glass',7:'hair',8:'leather',9:'metal',10:'mirror',11:'other',12:'painted',13:'paper',14:'plastic',15:'polishedstone',16:'skin',17:'sky',18:'stone',19:'tile',20:'wallpaper',21:'water',22:'wood'}


#Resize transform
resize_tr = Resize((SIZE, SIZE))
softmax = nn.Softmax(dim=0)

#load photo and resize
rgb = read_image(FILE[0]).cuda()
rgb = resize_tr(rgb)/255
rgb = torch.unsqueeze(rgb, 0)


if EXPERIMEMT >= 1:
    nir = read_image(FILE[1]).cuda()
    nir = resize_tr(nir)/255
    nir = torch.unsqueeze(nir, 0)

if EXPERIMEMT >= 2:
    dpt = read_image(FILE[2]).cuda()
    dpt = resize_tr(dpt)/255
    dpt = torch.unsqueeze(dpt, 0)


#inference
Y = -1
if EXPERIMEMT == -1:
    Y = model(rgb)
    Y = softmax(Y[0][:23])
    Y = torch.argmax(Y, 0).detach().cpu()
    Y = int(Y)
    print('This is :', material_minc[Y])

elif EXPERIMEMT in [0, 1, 2]:
    Y = model(rgb, nir, dpt)
    Y = softmax(Y[0][:23])
    Y = torch.argmax(Y, 0).detach().cpu()
    Y = int(Y)
    print('This is :', material_minc[Y])