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


LOAD = True
model, _, log_file, SIZE, BATCH_SIZE = model_params(model_name=sys.argv[1], load=LOAD).get() #"swinv2b", "vith14", "eva02l14", "maxvitxl", coatnet2
log_file = open(log_file, "a")
EPOCHS, LR  = 30, 4e-5
checkpoint = 'Material_recognition/weights/' + sys.argv[1] + '_irh.pth'

train_dataset = IRHDataset("Material_recognition/datasets/irh/files/img_raw", "Material_recognition/datasets/irh/dataset.csv", (SIZE, SIZE))
train_loader = DataLoader(dataset=train_dataset, batch_size=1, num_workers=4, pin_memory=False, shuffle=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-8)
lr_scheduler = CosineDecayLR(optimizer, LR, len(train_loader) * EPOCHS)
loss = nn.CrossEntropyLoss().cuda()

#One epoch done
for epc in range(0, EPOCHS):
    ticket = "Epoch {}: ".format(epc)
    log_file.write(ticket + "\n")
    
    r = tqdm((train_loader), leave=False, desc=ticket, total=len(train_loader))    
    for x, y in r:
        y_pred = model(x.cuda())
        lf = loss(y_pred, y.cuda())
        lf.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        optimizer.zero_grad()
        r.set_postfix(loss=lf.item())
    #decrease lr
    LR = lr_scheduler.step(epc * len(train_loader) + idx)
    torch.save(model.state_dict(), checkpoint)
log_file.close()
