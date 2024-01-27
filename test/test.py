import ast
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

from models.coatnet2_multimodal import coatnet_full


minc_path = 'Material_recognition/datasets/minc'
labels_path_t = 'Material_recognition/datasets/minc/test.txt'
checkpoint = "Material_recognition/weights/swinv2b_minc.pth"

#MODEL_NAME, NUM_WORKERS = sys.argv[1], int(sys.argv[2])
#model, _, _, SIZE, BATCH_SIZE = model_params(model_name=MODEL_NAME, load=True).get()


model, SIZE, BATCH_SIZE, NUM_WORKERS,  = coatnet_full(0), 384, 2, 4

dataloader = DataLoader(dataset=MINCDataset(minc_path, labels_path_t, size=(SIZE, SIZE)), 
            batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True, shuffle=True)

#add specific dataloaders list
#convert minc results. drop classes


#zero-shot accuracy. 
#minc accuracy dropping classes. 
#finetuned models test on minc. 
#finetuned models test on irh. 


#crf test.

#devide data ipynb test and train.




loss = Metrics()

#print("Model name:", MODEL_NAME)
print("Model name:", "Coatnet2 RGB IRH")
print("Num workers:", NUM_WORKERS)

with torch.no_grad():
        r = tqdm(enumerate(dataloader), leave=False, desc="Test: ", total=len(dataloader))
        ac1, ac5, i = 0, 0, 0
        try:
            for idx, (X, Y) in r:
                x_rgb, x_nir, x_dpt = X, torch.Tensor(0), torch.Tensor(0)
                y_pred = model(x_rgb.cuda())
                #l = loss.accuracies(y_pred, y)
                l = (loss.accuracy_irh(y_pred, Y), 99)

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
