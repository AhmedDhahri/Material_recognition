import sys
sys.path.append(sys.path[0]+'/../')
sys.path.append(sys.path[0]+'/../utils')

import timm
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataloaders.irh_dataloader import IRHDataset
from utils import CosineDecayLR, Metrics
from model_params import model_params
from timm.layers import SelectAdaptivePool2d

class classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.adapool = SelectAdaptivePool2d(pool_type='avg', flatten=nn.Flatten(start_dim=1, end_dim=-1))
    def forward(self, x, pre_logits):
        return self.adapool(x)

class coatnet_full(nn.Module):
    def __init__(self, experiment):
        super().__init__()
        self.experiment = experiment
        if experiment >= 0:
            self.bb_rgb = timm.create_model('coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k', pretrained=False)
            self.bb_rgb.load_state_dict(torch.load("Material_recognition/weights/coatnet2_minc.pth"), strict=False)
            self.bb_rgb.head = classifier()
            self.fc = nn.Linear(in_features=1024, out_features=15, bias=True)

        if experiment >= 1:
            self.bb_nir = timm.create_model('coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k', pretrained=False)
            self.bb_nir.load_state_dict(torch.load("Material_recognition/weights/coatnet2_minc.pth"), strict=False)
            self.bb_nir.head = classifier()
            self.fc = nn.Linear(in_features=2*1024, out_features=15, bias=True)

        if experiment >= 2:
            self.bb_dpt = timm.create_model('coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k', pretrained=False)
            self.bb_dpt.load_state_dict(torch.load("Material_recognition/weights/coatnet2_minc.pth"), strict=False)
            self.bb_dpt.head = classifier()
            self.fc = nn.Linear(in_features=3*1024, out_features=15, bias=True)

    def forward(self, x_rgb, x_nir, x_dpt):
        x_rgb = self.bb_rgb(x_rgb)
        
        if self.experiment >= 1:
            x_nir = self.bb_nir(x_nir)
            x_rgb = torch.cat((x_rgb, x_nir), -1)

        if self.experiment >= 2:
            x_dpt = self.bb_dpt(x_dpt)
            x_rgb = torch.cat((x_rgb, x_dpt), -1)
    
        return self.fc(x_rgb)

TRAIN_ITER = 500
BATCH_SIZE, EPOCHS, EXPERIMENT, LOAD = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), True
#model, _, log_file, SIZE, BATCH_SIZE = model_params(model_name=sys.argv[1], load=LOAD).get() #"swinv2b", "vith14", "eva02l14", "maxvitxl", coatnet2
SIZE, LR  = 384, 4e-5
model = coatnet_full(EXPERIMENT).cuda()

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
