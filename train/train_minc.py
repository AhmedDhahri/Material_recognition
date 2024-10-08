import ast
import sys
sys.path.append(sys.path[0]+'/../')
sys.path.append(sys.path[0]+'/../utils')

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataloaders.minc_dataloader import MINCDataset
from utils import CosineDecayLR, Metrics
from model_params import model_params

#use python 3.10- pip3 install torch torchvision tqdm timm
EPOCHS, LR = 5, 4e-5
minc_path = 'Material_recognition/datasets/minc'
labels_path = 'Material_recognition/datasets/minc/train.txt'
labels_path_t = 'Material_recognition/datasets/minc/test.txt'

MODEL_NAME, LOAD, NUM_WORKERS, START = sys.argv[1], ast.literal_eval(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
print("Loading pre-trained weights", MODEL_NAME, ":", LOAD)
print("Num workers:", NUM_WORKERS)
model, checkpoint, log_file_path, SIZE, BATCH_SIZE = model_params(model_name=MODEL_NAME, load=LOAD).get()
TRAIN_ITER, TEST_ITER  = 80000 // BATCH_SIZE, 8000 // BATCH_SIZE


train_loader = DataLoader(dataset=MINCDataset(minc_path, labels_path, size=(SIZE, SIZE)), 
                         batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True, shuffle=True)
test_loader = DataLoader(dataset=MINCDataset(minc_path, labels_path_t, size=(SIZE, SIZE)), 
                         batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True, shuffle=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-8)
lr_scheduler = CosineDecayLR(optimizer, LR, len(train_loader) * EPOCHS)
loss = Metrics()


#Training loop
for epc in range(START, EPOCHS):

    #Start log file
    log_file = open(log_file_path, "w+")
    ticket = "Epoch {}: ".format(epc)
    log_file.write(ticket + "\n")
    
    #Epoch loop
    r = tqdm(train_loader, leave=False, desc=ticket, total=len(train_loader))    
    for idx, (x, y) in enumerate(r):
        y_pred = model(x.cuda())
        lf = loss.compute(y_pred, y)
        lf.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        optimizer.zero_grad()
        r.set_postfix(loss=lf.item())
        
        if idx % TRAIN_ITER == 0 & idx != 0:
            #save checkpoint
            if idx != 0:
                torch.save(model.state_dict(), checkpoint)
            #decrease lr
            LR = lr_scheduler.step(epc * len(train_loader) + idx)
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
    torch.save(model.state_dict(), checkpoint)
    log_file.close()
