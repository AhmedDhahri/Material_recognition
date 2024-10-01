import ast
import sys
sys.path.append(sys.path[0]+'/../')
sys.path.append(sys.path[0]+'/../utils')

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataloaders.irh_dataloader import IRHDataset
from utils import Metrics

from model_params import model_params
from models.coatnet2_multimodal import coatnet_full




MODEL_NAME, EXPERIMENT, NUM_WORKERS, BATCH_SIZE = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])

#list of dataloaders specific to one class
if EXPERIMENT == 0:
    checkpoint = 'Material_recognition/weights/coatnet2_rgb_irh.pth' 
elif EXPERIMENT == 1:
    checkpoint = 'Material_recognition/weights/coatnet2_rgb_nir_irh.pth' 
elif EXPERIMENT == 2:
    checkpoint = 'Material_recognition/weights/coatnet2_full_irh.pth' 
model, SIZE = coatnet_full(EXPERIMENT, False), 384
model.load_state_dict(torch.load(checkpoint), strict=False)
model = model.cuda()

dataset = IRHDataset("Material_recognition/datasets/irh/img_raw", "Material_recognition/datasets/irh/dataset.csv", (SIZE, SIZE), EXPERIMENT)
loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True, shuffle=True)

loss = Metrics()

print("Model name:", MODEL_NAME)
print("Num workers:", NUM_WORKERS)

with torch.no_grad():
        r = tqdm(enumerate(loader), leave=False, desc="Test: ", total=len(loader))
        ac1, ac5, i = 0, 0, 0
        try:
            for idx, (X, Y) in r:
                if EXPERIMENT == 0:
                    x_rgb, x_nir, x_dpt = X, torch.Tensor(0), torch.Tensor(0)
                elif EXPERIMENT == 1:
                    (x_rgb, x_nir), x_dpt = X, torch.Tensor(0)
                elif EXPERIMENT == 2:
                    x_rgb, x_nir, x_dpt = X
                y_pred = model(x_rgb.cuda(), x_nir.cuda(), x_dpt.cuda())
                l = loss.accuracies(y_pred.item(), Y)
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
