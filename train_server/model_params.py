import torch
import torch.nn as nn

from torchvision.models import swin_v2_b, Swin_V2_B_Weights

log_file_path = 'Material_recognition/logs/swinv2b.log'
checkpoint = 'Material_recognition/weights/swin_v2b_minc.pth'