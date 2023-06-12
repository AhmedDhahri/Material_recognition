import sys
sys.path.append(sys.path[0]+'/../')
sys.path.append(sys.path[0]+'/../utils')

import torch
import torch.nn as nn
import time
from tqdm import tqdm
from torchsummary import summary
from torch.utils.data import DataLoader

from dataloaders.minc_dataloader import MINCDataset, MINCDataLoader
from utils import CosineDecayLR, Metrics

from torchvision.models import swin_v2_b, Swin_V2_B_Weights


log_file_path = 'Material_recognition/logs/swinv2b.log'
minc_path = 'Material_recognition/datasets/minc'
labels_path = 'Material_recognition/datasets/minc/train.txt'
labels_path_t = 'Material_recognition/datasets/minc/test.txt'
checkpoint = "Material_recognition/weights/swin_v2b_minc.pth"

BATCH_SIZE = 8
TRAIN_ITER = 2000
TEST_ITER = 400
size = 256

LOAD = False
lr = 4e-5
start = 0
EPOCHS = 30



log_file = open(log_file_path, "w+")
train_loader = MINCDataLoader(minc_path, labels_path, batch_size=BATCH_SIZE, size=size, f=0.16)
test_loader = DataLoader(dataset=MINCDataset(minc_path, labels_path_t, size=(size, size)), 
                         batch_size=BATCH_SIZE, num_workers=0, pin_memory=False, shuffle=True)

model = swin_v2_b(weights=Swin_V2_B_Weights.IMAGENET1K_V1)

if LOAD:
    model.load_state_dict(torch.load(checkpoint), strict=False)
model = model.train()
model = model.cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-8)
lr_scheduler = CosineDecayLR(optimizer, lr, len(train_loader) * EPOCHS)
loss = Metrics()

#One epoch done
for epc in range(0, EPOCHS):
    ticket = "Epoch {} starting from iteration {}: ".format(epc, start)
    log_file.write(ticket + "\n")
    
    r = tqdm(range(start, len(train_loader)), leave=False, desc=ticket, total=len(train_loader)-start)    
    for idx in r:
        x, y = train_loader[idx]
        y_pred = model(x.cuda())
        lf = loss.compute(y_pred, y)
        lf.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        optimizer.zero_grad()
        r.set_postfix(loss=lf.item())
        
        if idx % TRAIN_ITER == 0:
            #save checkpoint
            if idx != start:
                torch.save(model.state_dict(), checkpoint)
            #decrease lr
            lr = lr_scheduler.step(epc * len(train_loader) + start + idx)
            #test loss run on val
            with torch.no_grad():
                ac = 0
                for i in range(TEST_ITER):
                    x, y = next(iter(test_loader))
                    y_pred = model(x.cuda())
                    ac = ac + loss.accuracy(y_pred, y)
                ac = ac/TEST_ITER
                log = "Iteration:", idx,
                    "\nTest N", str(epc) + '-' + str(idx//TRAIN_ITER) + ' : ' + str(float(ac)),
                     "\nLearning rate:", lr,
                print(log)
                log_file.write(log+ "\n")
                log_file.flush()
			
    start = 0
    torch.save(model.state_dict(), checkpoint)
    log_file.close()
