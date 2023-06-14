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


minc_path = 'Material_recognition/datasets/minc'
labels_path = 'Material_recognition/datasets/minc/train.txt'
labels_path_t = 'Material_recognition/datasets/minc/test.txt'

BATCH_SIZE, EPOCHS, SIZE, LR = 8, 30, 256, 4e-5
TRAIN_ITER, TEST_ITER, START, LOAD = 2000, 400, 0, False



train_loader = MINCDataLoader(minc_path, labels_path, batch_size=BATCH_SIZE, size=SIZE, f=0.16)
test_loader = DataLoader(dataset=MINCDataset(minc_path, labels_path_t, size=(SIZE, SIZE)), 
                         batch_size=BATCH_SIZE, num_workers=0, pin_memory=False, shuffle=True)

model, checkpoint, log_file = model_params(model_name=sys.argv[1], load=LOAD).get() #"swinv2b", "vith14", "eva02l14", "maxvitxl"

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-8)
lr_scheduler = CosineDecayLR(optimizer, LR, len(train_loader) * EPOCHS)
loss = Metrics()

#One epoch done
for epc in range(0, EPOCHS):
    ticket = "Epoch {} starting from iteration {}: ".format(epc, START)
    log_file.write(ticket + "\n")
    
    r = tqdm(range(START, len(train_loader)), leave=False, desc=ticket, total=len(train_loader)-START)    
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
                log = "Iteration:" + idx + "\nTest N"+ str(epc) + '-' + str(idx//TRAIN_ITER) + ' : '  +  str(float(ac)) + "\nLearning rate:" + LR
                print(log)
                log_file.write(log+ "\n")
                log_file.flush()
			
    START = 0
    torch.save(model.state_dict(), checkpoint)
    log_file.close()
