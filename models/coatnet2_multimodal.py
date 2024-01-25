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