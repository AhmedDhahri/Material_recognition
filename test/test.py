import sys
sys.path.append(sys.path[0]+'/../')
sys.path.append(sys.path[0]+'/../utils')

import timm
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataloaders.minc_dataloader import MINCDataset
from dataloaders.irh_dataloader import IRHDataset

from utils import Metrics
from models.coatnet2_multimodal import coatnet_full

CLS, EXPERIEMT = int(sys.argv[1]), int(sys.argv[2])# irh 0,1,2; minc 3

SIZE, BATCH_SIZE, NUM_WORKERS, model, dataloader = 384, 8, 8, 0, 0

minc_path = 'Material_recognition/datasets/minc'
minc_labels_t = 'Material_recognition/datasets/minc/test.txt'
minc_checkpoint = "Material_recognition/weights/coatnet2_minc.pth"

irh_path = "Material_recognition/datasets/irh/img_raw"
irh_labels = "Material_recognition/datasets/irh/dataset.csv"
irh_checkpoint = ""
if EXPERIEMT == 0 or EXPERIEMT == 4:
    irh_checkpoint = "Material_recognition/weights/coatnet2_rgb_irh.pth"
elif EXPERIEMT == 1:
    irh_checkpoint = "Material_recognition/weights/coatnet2_rgb_nir_irh.pth"
elif EXPERIEMT == 2:
    irh_checkpoint = "Material_recognition/weights/coatnet2_full_irh.pth"

#MODEL_NAME, NUM_WORKERS = sys.argv[1], int(sys.argv[2])
#model, _, _, SIZE, BATCH_SIZE = model_params(model_name=MODEL_NAME, load=True).get()


if EXPERIEMT in [0,1,2]:
    model = coatnet_full(EXPERIEMT, load=False).cuda()
    model.load_state_dict(torch.load(irh_checkpoint), strict=False)
    dataloader = DataLoader(IRHDataset(irh_path, irh_labels, (SIZE, SIZE), experiment=EXPERIEMT, cls=CLS), 
            batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True, shuffle=True)
elif EXPERIEMT == 3:
    dataloader = DataLoader(dataset=MINCDataset(minc_path, minc_labels_t, size=(SIZE, SIZE), cls=CLS), 
            batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True, shuffle=True)
    model = timm.create_model('coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k', pretrained=False)
    model.load_state_dict(torch.load(minc_checkpoint), strict=False)
elif EXPERIEMT == 4:
    model = coatnet_full(0, load=False).cuda()
    model.load_state_dict(torch.load(irh_checkpoint), strict=False)
    dataloader = DataLoader(IRHDataset(irh_path, irh_labels, (SIZE, SIZE), experiment=0, cls=CLS), 
            batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True, shuffle=True)



model = model.eval()
model = model.cuda()


#devide data ipynb test and train.
#zero-shot accuracy. 
#add specific dataloaders list and test per class

#finetuned models test on minc. -> 60 but improving
#finetuned models test on irh.  -> 90 but verify after improving the 60
#crf test.  -> after improving the 60



#convert minc results. drop classes -> done 90.27


loss = Metrics()

with torch.no_grad():
        r = tqdm(enumerate(dataloader), leave=False, desc="Test: ", total=len(dataloader))
        ac1, ac5, i = 0, 0, 0
        try:
            for idx, (X, Y) in r:
                x_rgb, x_nir, x_dpt, y_pred, l = 0,0,0,0,0
                if EXPERIEMT == 0:
                    x_rgb, x_nir, x_dpt = X, torch.Tensor(0), torch.Tensor(0)
                    y_pred = model(x_rgb.cuda(), x_nir.cuda(), x_dpt.cuda())
                    l = (loss.accuracy(y_pred, Y), 0.99)
                elif EXPERIEMT == 1:
                    (x_rgb, x_nir), x_dpt = X, torch.Tensor(0)
                    y_pred = model(x_rgb.cuda(), x_nir.cuda(), x_dpt.cuda())
                    l = (loss.accuracy(y_pred, Y), 0.99)
                elif EXPERIEMT == 2:
                    x_rgb, x_nir, x_dpt = X
                    y_pred = model(x_rgb.cuda(), x_nir.cuda(), x_dpt.cuda())
                    l = (loss.accuracy(y_pred, Y), 0.99)
                elif EXPERIEMT == 3:
                    y_pred = model(X.cuda())
                    l = (loss.accuracy_irh(y_pred, Y, False), 0.99)
                elif EXPERIEMT == 4:
                    x_rgb, x_nir, x_dpt = X, torch.Tensor(0), torch.Tensor(0)
                    y_pred = model(x_rgb.cuda(), x_nir.cuda(), x_dpt.cuda())
                    l = (loss.accuracy_irh(y_pred, Y, True), 0.99)
                    
                #l = loss.accuracies(y_pred, Y)
                

                ac1 += l[0]
                ac5 += l[1]
                r.set_postfix(loss=ac1/(idx+1))
                i = idx + 1

            ac1 /= i
            ac5 /= i
            print("Top 1 acccuracy ", ac1 * 100, "Top 5 error ", ac5 * 100)
        except KeyboardInterrupt:
            ac1 /= i
            ac5 /= i
            print("Top 1 acccuracy ", ac1 * 100, "Top 5 error ", ac5 * 100)
