import sys
sys.path.append(sys.path[0]+'/../')
sys.path.append(sys.path[0]+'/../utils')

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataloaders.minc_dataloader import MINCDataset, MINCDataLoader
from utils import CosineDecayLR, Metrics
from model_params import model_params


LOAD = False
model, checkpoint, log_file, SIZE, BATCH_SIZE = model_params(model_name=sys.argv[1], load=LOAD).get() #"swinv2b", "vith14", "eva02l14", "maxvitxl"
EPOCHS, LR  = 10, 4e-5

dataset = IRHDataset("../datasets/irh/files/img_raw", "../datasets/irh/dataset.csv", (256, 256))
loader = DataLoader(dataset=dataset, batch_size=4, num_workers=0, pin_memory=False, shuffle=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-8)
lr_scheduler = CosineDecayLR(optimizer, LR, len(train_loader) * EPOCHS)
loss = Metrics()

#One epoch done
for epc in range(0, EPOCHS):
    ticket = "Epoch {} starting from iteration {}: ".format(epc, START)
    log_file.write(ticket + "\n")
    
    r = tqdm(loader), leave=False, desc=ticket, total=len(loader))    
    for idx, x, y in r:
        y_pred = model(x.cuda())
        lf = loss.compute(y_pred, y)
        lf.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        optimizer.zero_grad()
        r.set_postfix(loss=lf.item())
        
        if idx % TRAIN_ITER == 0:
            #save checkpoint
            if idx != START:
                torch.save(model.state_dict(), checkpoint)
            #decrease lr
            LR = lr_scheduler.step(epc * len(train_loader) + START + idx)
            #test loss run on val
            with torch.no_grad():
                ac = 0
                for i in range(TEST_ITER):
                    x, y = next(iter(test_loader))
                    y_pred = model(x.cuda())
                    ac = ac + loss.accuracy(y_pred, y)
                ac = ac/TEST_ITER
                log = "Iteration:" + str(idx) + "\nTest N"+ str(epc) + '-' + str(idx//TRAIN_ITER) + ' : '  +  str(float(ac)) + "\nLearning rate:" + str(LR)
                print(log)
                log_file.write(log+ "\n")
                log_file.flush()
			
    START = 0
    torch.save(model.state_dict(), checkpoint)
    log_file.close()
