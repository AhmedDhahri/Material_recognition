import sys
sys.path.append(sys.path[0]+'/../')
sys.path.append(sys.path[0]+'/../utils')

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataloaders.minc_dataloader import MINCDataset
from utils import Metrics

from torchvision.models import swin_v2_b
from model_params import model_params



minc_path = 'Material_recognition/datasets/minc'
labels_path_t = 'Material_recognition/datasets/minc/test.txt'
checkpoint = "Material_recognition/weights/swinv2b_minc.pth"

MODEL_NAME, NUM_WORKERS = sys.argv[1], ast.literal_eval(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
dataloader = DataLoader(dataset=MINCDataset(minc_path, labels_path_t, size=(size, size)), 
            batch_size=16, num_workers=NUM_WORKERS, pin_memory=False, shuffle=True)
model, _, _, SIZE, BATCH_SIZE = model_params(model_name=MODEL_NAME, load=True).get()
loss = Metrics()

print("Model name:", MODEL_NAME)
print("Num workers:", NUM_WORKERS)

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
            print("Top 1 acccuracy ", ac1 * 100, "Top 5 error ", ac5 * 100)
        except KeyboardInterrupt:
            ac1 /= i
            ac5 /= i
            print("Top 1 acccuracy ", ac1 * 100, "Top 5 error ", ac5 * 100)
