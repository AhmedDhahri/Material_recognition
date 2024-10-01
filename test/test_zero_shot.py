import sys
sys.path.append(sys.path[0]+'/../')
sys.path.append(sys.path[0]+'/../utils')

import timm
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataloaders.minc_dataloader import MINCDataset
from dataloaders.irh_dataloader import IRHDataset

from utils import Metrics
from models.coatnet2_multimodal import coatnet_full