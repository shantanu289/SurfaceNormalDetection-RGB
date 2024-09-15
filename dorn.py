import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models
import collections
import math


def init_weights(modules):
    m = modules
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)
        m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        m.bias.data.zero_()


class ImageEncoder(nn.Module):
    def __init__(self):
      super(ImageEncoder, self).__init__()
      self.global_pooling = nn.AvgPool2d(8, stride=8, padding=(1, 0))
      self.dropout = nn.Dropout2d(p=0.5)
      self.global_fc = nn.Linear(2048*4*5, 512)
      self.relu = nn.ReLU(inplace=True)
      self.conv1 = nn.Conv2d(512, 512, 1)
      self.upsample = nn.UpsamplingBilinear2d(size=(30, 40))

      nn.init.xavier_normal_(self.conv1.weight)
      nn.init.xavier_normal_(self.global_fc.weight)

    def forward(self, x):
        x1 = self.global_pooling(x)
        x2 = self.dropout(x1)
        x3 = x2.view(-1, 2048*4*5)
        x3 = self.global_fc(x3)
        x4 = self.relu(x3)
        x4 = x4.view(-1,512,1,1)
        x5 = self.conv1(x4)
        x6 = self.upsample(x5)
        return x6

class ResNet(nn.Module):
    def __init__(self, input_channels, pretrained=True, freeze=True):
        super(ResNet, self).__init__()
        pretrained_model = torchvision.models.__dict__['resnet{}'.format(101)](pretrained=pretrained)

        self.channels = input_channels

        self.conv1_1 = nn.Conv2d(self.channels, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(64)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_2 = nn.BatchNorm2d(64)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.conv1_3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_3 = nn.BatchNorm2d(128)
        self.relu1_3 = nn.ReLU(inplace=True)

        self.bn1 = nn.BatchNorm2d(128)

        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']
        self.layer1 = pretrained_model._modules['layer1']
        self.layer1[0].conv1 = nn.Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer1[0].downsample[0] = nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer2 = pretrained_model._modules['layer2']
        self.layer3 = pretrained_model._modules['layer3']
        self.layer3[0].conv2 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer3[0].downsample[0] = nn.Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer4 = pretrained_model._modules['layer4']
        self.layer4[0].conv2 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer4[0].downsample[0] = nn.Conv2d(1024, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)

        #initialize weights
        nn.init.kaiming_normal_(self.conv1_1.weight)
        self.bn_1.weight.data.fill_(1.0)
        nn.init.kaiming_normal_(self.conv1_2.weight)
        self.bn_2.weight.data.fill_(1.0)
        nn.init.kaiming_normal_(self.conv1_3.weight)
        self.bn1_3.weight.data.fill_(1.0)
        self.bn1.weight.data.fill_(1.0)
        nn.init.kaiming_normal_(self.layer1[0].conv1.weight)
        nn.init.kaiming_normal_(self.layer1[0].downsample[0].weight)
        nn.init.kaiming_normal_(self.layer3[0].conv2.weight)
        nn.init.kaiming_normal_(self.layer3[0].downsample[0].weight)
        nn.init.kaiming_normal_(self.layer4[0].conv2.weight)
        nn.init.kaiming_normal_(self.layer4[0].downsample[0].weight)

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.bn_1(x)
        x = self.relu1_1(x)
        x = self.conv1_2(x)
        x = self.bn_2(x)
        x = self.relu1_2(x)
        x = self.conv1_3(x)
        x = self.bn1_3(x)
        x = self.relu1_3(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return x4

class SceneUnderstandingModule(nn.Module):
    def __init__(self, output_channels=136):
        super(SceneUnderstandingModule, self).__init__()
        self.encoder = ImageEncoder()
        self.aspp1 = nn.Sequential(
            nn.Conv2d(2048, 512, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,1),
            nn.ReLU(inplace=True)
        )
        self.aspp2 = nn.Sequential(
            nn.Conv2d(2048, 512, 3, padding=6, dilation=6),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,1),
            nn.ReLU(inplace=True)
        )
        self.aspp3 = nn.Sequential(
            nn.Conv2d(2048, 512, 3, padding=12, dilation=12),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,1),
            nn.ReLU(inplace=True)
        )
        self.aspp4 = nn.Sequential(
            nn.Conv2d(2048, 512, 3, padding=18, dilation=18),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 1),
            nn.ReLU(inplace=True)
        )
        self.concatenate = nn.Sequential(
            nn.Dropout2d(p=0.5),
            nn.Conv2d(512*5, 2048, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
            nn.Conv2d(2048, output_channels, 1),
            nn.UpsamplingBilinear2d(size=(240,320))
        )
        init_weights(self.modules())

    def forward(self, x):
        
        x1 = self.encoder(x)
        x2 = self.aspp1(x)
        x3 = self.aspp2(x)
        x4 = self.aspp3(x)
        x5 = self.aspp4(x)
        x6 = torch.cat((x1, x2, x3, x4, x5), dim=1)        

        xout = self.concatenate(x6)
        return xout

class DORN(nn.Module):
    def __init__(self, output_size=(240,320), loss_function=1, channels=5, pretrained=True, output_channels=13):
        super(DORN, self).__init__()
        self.loss_function = loss_function
        self.output_size = output_size
        self.channels = channels
        self.feature_extractor = ResNet(channels, pretrained)
        self.aspp_module = SceneUnderstandingModule(output_channels)


    def forward(self, x):
        x1 = self.feature_extractor(x)        
        x2 = self.aspp_module(x1)
        return x2




    