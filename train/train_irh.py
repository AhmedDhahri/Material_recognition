import sys
sys.path.append(sys.path[0]+'/../')
sys.path.append(sys.path[0]+'/../utils')

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataloaders.irh_dataloader import IRHDataset
from utils import CosineDecayLR, Metrics
from model_params import model_params
from models.coatnet2_multimodal import coatnet_full


TRAIN_ITER = 500
BATCH_SIZE, EPOCHS, EXPERIMENT, LOAD = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), True
#model, _, log_file, SIZE, BATCH_SIZE = model_params(model_name=sys.argv[1], load=LOAD).get() #"swinv2b", "vith14", "eva02l14", "maxvitxl", coatnet2
SIZE, LR  = 384, 4e-5
model = coatnet_full(EXPERIMENT)
model = model.eval()
model = model.cuda()

global net_name 
if EXPERIMENT == 0:
    net_name = 'coatnet2_rgb_irh' 
elif EXPERIMENT == 1:
    net_name = 'coatnet2_rgb_nir_irh' 
elif EXPERIMENT == 2:
    net_name = 'coatnet2_full_irh' 
else:
    print(EXPERIMENT, "is invalid.")
    exit(0)

checkpoint, log_file = 'Material_recognition/weights/' + net_name + '.pth', open('Material_recognition/logs/' + net_name + '.log', "a")
train_dataset = IRHDataset("Material_recognition/datasets/irh/files/img_raw", "Material_recognition/datasets/irh/dataset.csv", (SIZE, SIZE), EXPERIMENT)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, num_workers=24, pin_memory=True, shuffle=True)
test_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, num_workers=24, pin_memory=True, shuffle=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-8)
lr_scheduler = CosineDecayLR(optimizer, LR, len(train_loader) * EPOCHS)
loss = nn.CrossEntropyLoss().cuda()
metric = Metrics()
#One epoch done
for epc in range(0, EPOCHS):
    ticket = "Epoch {}: ".format(epc)
    r = tqdm((train_loader), leave=False, desc=ticket, total=len(train_loader))    
    for X, Y in r:
        if EXPERIMENT == 0:
            x_rgb, x_nir, x_dpt = X, torch.Tensor(0), torch.Tensor(0)
        elif EXPERIMENT == 1:
            (x_rgb, x_nir), x_dpt = X, torch.Tensor(0)
        elif EXPERIMENT == 2:
            x_rgb, x_nir, x_dpt = X

        Y_pred = model(x_rgb.cuda(), x_nir.cuda(), x_dpt.cuda())
        lf = loss(Y_pred, Y.cuda())
        lf.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        optimizer.zero_grad()
        r.set_postfix(loss=lf.item())
    #decrease lr
    LR = lr_scheduler.step(epc * len(train_loader))
    with torch.no_grad():
        ac = 0
        for i in range(TRAIN_ITER):
            X, Y = next(iter(test_loader))
            if EXPERIMENT == 0:
                x_rgb, x_nir, x_dpt = X, torch.Tensor(0), torch.Tensor(0)
            elif EXPERIMENT == 1:
                (x_rgb, x_nir), x_dpt = X, torch.Tensor(0)
            elif EXPERIMENT == 2:
                x_rgb, x_nir, x_dpt = X
            Y_pred = model(x_rgb.cuda(), x_nir.cuda(), x_dpt.cuda())
            ac += metric.accuracy(Y_pred, Y)
        log = "Accuracy {} Learning rate: {}\n".format(ac/TRAIN_ITER, LR)
        print(log)
        log_file.write(ticket + log)
        log_file.flush()
    torch.save(model.state_dict(), checkpoint)
log_file.close()
