
import sys
sys.path.append(sys.path[0]+'/../')

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataloaders.minc_dataloader import MINCDataset_gnet
from utils.loss_resnet import LossRN

from models.googlenet import googlenet as Gnet



minc_path = '/home/ahmed/workspace/notebook/matrec/datasets/minc'
labels_path_t = '/home/ahmed/workspace/notebook/matrec/datasets/minc/test.txt'
BATCH_SIZE = 4
size = 256


checkpoint = "../weights/minc-googlenet.pth"

dataset = MINCDataset_gnet(minc_path, labels_path_t, size=(size, size))
dataloader = DataLoader(dataset=dataset, batch_size=16, num_workers=0, pin_memory=False, shuffle=True)


model = Gnet().cuda()
model.load_state_dict(torch.load(checkpoint), strict=False)

model = model.eval()
loss = LossRN()

with torch.no_grad():
        r = tqdm(enumerate(dataloader), leave=False, desc="Test: ", total=len(dataloader))
        ac1, ac5, i = 0, 0, 0
        try:
            for idx, (x, y) in r:
                y_pred = model(x.cuda())
                l = loss.accuracies(y_pred, y)
                ac1 += l[0]
                ac5 += l[1]
                r.set_postfix(loss=ac1/(idx+1))
                i = idx + 1

            ac1 /= i
            ac5 /= i
            print("Top 1 error ", ac1 * 100, "Top 5 error ", ac5 * 100)
        except KeyboardInterrupt:
            ac1 /= i
            ac5 /= i
            print("Top 1 error ", ac1 * 100, "Top 5 error ", ac5 * 100)
