import torch
import torch.nn as nn


class BasicBlock(nn.Module):
  def __init__(self, in_planes, out_planes, stride):
    super().__init__()
    self.flag = False
    self.downsample = None

    self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(out_planes)

    self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(out_planes)

    if stride != 1 or in_planes != out_planes:
      self.flag = True
      self.downsample = nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                                nn.BatchNorm2d(out_planes))

    self.relu = torch.nn.ReLU(inplace=True)

  def forward(self, x):
    out = self.relu(self.bn1(self.conv1(x)))
    out = self.bn2(self.conv2(out))
    if self.flag:
      x = self.shortcut_bn(self.shortcut_conv(x))
    out += x
    out = self.relu(out)

    return out

def ResBlock(in_planes, out_planes, stride, num_blocks):
  sequential = []
  sequential.append(BasicBlock(in_planes, out_planes, stride))
  for i in range(1,num_blocks):
    sequential.append(BasicBlock(out_planes, out_planes, 1))


  return nn.Sequential(*sequential)

class ResNet34(nn.Module):
  def __init__(self):
    super(ResNet34, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    #when backboning this layer is droped
    self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2)

    self.layer1 = ResBlock(64, 64, 1, 3)
    self.layer2 = ResBlock(64, 128, 2, 4)
    self.layer3 = ResBlock(128, 256, 2, 6)
    self.layer4 = ResBlock(256, 512, 2, 3)


    self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    self.fc = nn.Linear(512, 1000)
    self.relu = torch.nn.ReLU(inplace=True)

  def forward(self, x):
    x = self.relu(self.bn1(self.conv1(x)))
    x = self.max_pool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avg_pool(x)
    x = torch.squeeze(x)
    x = self.fc(x)

    return x


class ResNet18(nn.Module):
  def __init__(self):
    super(ResNet18, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    #when backboning this layer is droped
    self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2)

    self.layer1 = ResBlock(64, 64, 1, 2)
    self.layer2 = ResBlock(64, 128, 2, 2)
    self.layer3 = ResBlock(128, 256, 2, 2)
    self.layer4 = ResBlock(256, 512, 2, 2)


    self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    self.fc = nn.Linear(512, 1000)
    self.relu = torch.nn.ReLU(inplace=True)

  def forward(self, x):
    x = self.relu(self.bn1(self.conv1(x)))
    x = self.max_pool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avg_pool(x)
    x = torch.squeeze(x)
    x = self.fc(x)

    return x
