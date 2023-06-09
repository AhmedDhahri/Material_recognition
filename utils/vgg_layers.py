from torchvision.models import vgg16
from torchvision.models import VGG16_Weights
import torch
import torch.nn as nn

class VGG16(nn.Module):
  def __init__(self):
    super(VGG16, self).__init__()
    self.conv1_1 = nn.Conv2d(3, 64, 3, 1, 1)
    self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 1)

    self.conv2_1 = nn.Conv2d(64, 128, 3, 1, 1)
    self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 1)

    self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 1)
    self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 1)
    self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 1)

    self.conv4_1 = nn.Conv2d(256, 512, 3, 1, 1)
    self.conv4_2 = nn.Conv2d(512, 512, 3, 1, 1)
    self.conv4_3 = nn.Conv2d(512, 512, 3, 1, 1)

    self.conv5_1 = nn.Conv2d(512, 512, 3, 1, 1)
    self.conv5_2 = nn.Conv2d(512, 512, 3, 1, 1)
    self.conv5_3 = nn.Conv2d(512, 512, 3, 1, 1)

    self.relu = nn.ReLU()
    self.pooling = nn.MaxPool2d(2, 2)

  def forward(self, x_0):
    x1_1 = self.relu(self.conv1_1(x_0))
    x1_2 = self.relu(self.conv1_2(x1_1))

    x2_0 = self.pooling(x1_2)

    x2_1 = self.relu(self.conv1_1(x2_0))
    x2_2 = self.relu(self.conv1_2(x2_1))

    x3_0 = self.pooling(x2_2)

    x3_1 = self.relu(self.conv1_1(x3_0))
    x3_2 = self.relu(self.conv1_2(x3_1))
    x3_3 = self.relu(self.conv1_2(x3_2))

    x4_0 = self.pooling(x3_3)

    x4_1 = self.relu(self.conv1_1(x4_0))
    x4_2 = self.relu(self.conv1_2(x4_1))
    x4_3 = self.relu(self.conv1_2(x4_2))

    x5_0 = self.pooling(x4_3)

    x5_1 = self.relu(self.conv1_1(x5_0))
    x5_2 = self.relu(self.conv1_2(x5_1))
    x5_3 = self.relu(self.conv1_2(x5_2))

    return x1_1, x1_2, x2_1, x2_2, x3_1, x3_2, x3_3, x4_1, x4_2, x4_3, x5_1, x5_2, x5_3

    def get_vgg_layer_model(self):
        params = self.state_dict()
        for idx, (n, w) in enumerate(vgg16(pretrained=True).named_parameters()):
          if 'features' in n:
            key = list(params.keys())[idx]
            params[key] = w
        self.load_state_dict(params)

class VGG16(nn.Module):
  def __init__(self):
    super(VGG16, self).__init__()
    self.conv1_1 = nn.Conv2d(3, 64, 3, 1, 1)
    self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 1)

    self.conv2_1 = nn.Conv2d(64, 128, 3, 1, 1)
    self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 1)

    self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 1)
    self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 1)
    self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 1)

    self.conv4_1 = nn.Conv2d(256, 512, 3, 1, 1)
    self.conv4_2 = nn.Conv2d(512, 512, 3, 1, 1)
    self.conv4_3 = nn.Conv2d(512, 512, 3, 1, 1)

    self.conv5_1 = nn.Conv2d(512, 512, 3, 1, 1)
    self.conv5_2 = nn.Conv2d(512, 512, 3, 1, 1)
    self.conv5_3 = nn.Conv2d(512, 512, 3, 1, 1)

    self.relu = nn.ReLU()
    self.pooling = nn.MaxPool2d(2, 2)

  def forward(self, x_0):
    x1_1 = self.relu(self.conv1_1(x_0))
    x1_2 = self.relu(self.conv1_2(x1_1))

    x2_0 = self.pooling(x1_2)

    x2_1 = self.relu(self.conv2_1(x2_0))
    x2_2 = self.relu(self.conv2_2(x2_1))

    x3_0 = self.pooling(x2_2)

    x3_1 = self.relu(self.conv3_1(x3_0))
    x3_2 = self.relu(self.conv3_2(x3_1))
    x3_3 = self.relu(self.conv3_3(x3_2))

    x4_0 = self.pooling(x3_3)

    x4_1 = self.relu(self.conv4_1(x4_0))
    x4_2 = self.relu(self.conv4_2(x4_1))
    x4_3 = self.relu(self.conv4_3(x4_2))

    x5_0 = self.pooling(x4_3)

    x5_1 = self.relu(self.conv5_1(x5_0))
    x5_2 = self.relu(self.conv5_2(x5_1))
    x5_3 = self.relu(self.conv5_3(x5_2))

    return x1_1, x1_2, x2_1, x2_2, x3_1, x3_2, x3_3, x4_1, x4_2, x4_3, x5_1, x5_2, x5_3

  def load_weights(self):

    params = self.state_dict()
    for idx, (n, w) in enumerate(vgg16(pretrained=True).named_parameters()):
      if 'features' in n:
        key = list(params.keys())[idx]
        params[key] = w
    self.load_state_dict(params)
class VGG16(nn.Module):
  def __init__(self):
    super(VGG16, self).__init__()
    self.conv1_1 = nn.Conv2d(3, 64, 3, 1, 1)
    self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 1)

    self.conv2_1 = nn.Conv2d(64, 128, 3, 1, 1)
    self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 1)

    self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 1)
    self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 1)
    self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 1)

    self.conv4_1 = nn.Conv2d(256, 512, 3, 1, 1)
    self.conv4_2 = nn.Conv2d(512, 512, 3, 1, 1)
    self.conv4_3 = nn.Conv2d(512, 512, 3, 1, 1)

    self.conv5_1 = nn.Conv2d(512, 512, 3, 1, 1)
    self.conv5_2 = nn.Conv2d(512, 512, 3, 1, 1)
    self.conv5_3 = nn.Conv2d(512, 512, 3, 1, 1)

    self.relu = nn.ReLU()
    self.pooling = nn.MaxPool2d(2, 2)

  def forward(self, x_0):
    x1_1 = self.relu(self.conv1_1(x_0))
    x1_2 = self.relu(self.conv1_2(x1_1))

    x2_0 = self.pooling(x1_2)

    x2_1 = self.relu(self.conv2_1(x2_0))
    x2_2 = self.relu(self.conv2_2(x2_1))

    x3_0 = self.pooling(x2_2)

    x3_1 = self.relu(self.conv3_1(x3_0))
    x3_2 = self.relu(self.conv3_2(x3_1))
    x3_3 = self.relu(self.conv3_3(x3_2))

    x4_0 = self.pooling(x3_3)

    x4_1 = self.relu(self.conv4_1(x4_0))
    x4_2 = self.relu(self.conv4_2(x4_1))
    x4_3 = self.relu(self.conv4_3(x4_2))

    x5_0 = self.pooling(x4_3)

    x5_1 = self.relu(self.conv5_1(x5_0))
    x5_2 = self.relu(self.conv5_2(x5_1))
    x5_3 = self.relu(self.conv5_3(x5_2))

    return x1_1, x1_2, x2_1, x2_2, x3_1, x3_2, x3_3, x4_1, x4_2, x4_3, x5_1, x5_2, x5_3

  def load_weights(self):

    params = self.state_dict()
    for idx, (n, w) in enumerate(vgg16(weights=VGG16_Weights.IMAGENET1K_V1).named_parameters()):
      if 'features' in n:
        key = list(params.keys())[idx]
        params[key] = w
    self.load_state_dict(params)


class Perceptual_Loss:
  def __init__(self):
    self.vgg = VGG16().cuda()
    self.vgg.load_weights()
    e = self.vgg.eval()

  def compute(self, x1, x2):
    loss = 0
    x1 = self.vgg(x1)
    x2 = self.vgg(x2)
    for i in range(int(len(x1)//2), len(x1)):
      loss += torch.nn.MSELoss()(x1[i], x2[i])
    return loss
