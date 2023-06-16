import timm
import torch
import torch.nn as nn

from torchvision.models import swin_v2_b, Swin_V2_B_Weights
from torchvision.models import vit_h_14, ViT_H_14_Weights


class model_params:
    def __init__(self, model_name='swinv2b', load=False):
        if model_name == 'swinv2b':
            if load:
                self.model = swin_v2_b()
            else:
                self.model = swin_v2_b(weights=Swin_V2_B_Weights.IMAGENET1K_V1)
            self.size, self.bs = 256, 8
        elif  model_name == 'vith14':
            if load:
                self.model = vit_h_14()
            else:
                self.model = vit_h_14(weights=ViT_H_14_Weights.IMAGENET1K_SWAG_LINEAR_V1)
            self.size, self.bs = 224, 4
        elif  model_name == 'eva02l14':
            self.model = timm.create_model('eva02_large_patch14_448.mim_m38m_ft_in1k', pretrained=not load)
            self.size, self.bs = 448, 4
        elif  model_name == 'maxvitxl':
            self.model = timm.create_model('maxvit_xlarge_tf_512.in21k_ft_in1k', pretrained=not load)
            self.size, self.bs = 512, 2
        elif model_name == 'coatnet2':
            self.model = timm.create_model('coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k', pretrained=not load)
            self.size, self.bs = 224, 8
        else:
            raise Exception("Sorry, model_name not valid!")
            exit()
        self.log_file_path = 'Material_recognition/logs/' + model_name + '.log'
        self.log_file = open(self.log_file_path, "w+")

        self.checkpoint = 'Material_recognition/weights/' + model_name + '_minc.pth'

        
        if load:
            self.model.load_state_dict(torch.load(self.checkpoint), strict=False)
        self.model = self.model.eval()
        self.model = self.model.cuda()

    def get(self):
        return self.model, self.checkpoint, self.log_file, self.size, self.bs