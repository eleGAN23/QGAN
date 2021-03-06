# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 02:12:33 2020

@author: Edoardo
"""

import sys
sys.path.append('../utils')

from utils.Ops import conv2d, Residual_G, Residual_D, spectral_norm, First_Residual_D, SNLinear
import torch
import torch.nn as nn
from torch.nn import BatchNorm2d


class SNGAN_D128(nn.Module):
    def __init__(self, spectral_normed=False, channel=3):
        super(SNGAN_D128, self).__init__()

        self.spectral_normed = spectral_normed
        # self.batch_normed = batch_normed
        
        self.re1 = First_Residual_D(channel, 64, spectral_normed = spectral_normed)
        self.re2 = Residual_D(64, 128, down_sampling = True, spectral_normed = spectral_normed)
        self.re3 = Residual_D(128, 256, down_sampling = True, spectral_normed = spectral_normed)
        self.re4 = Residual_D(256, 512, down_sampling = True, spectral_normed = spectral_normed)
        self.re5 = Residual_D(512, 1024, down_sampling = True, spectral_normed = spectral_normed)
        self.re6 = Residual_D(1024, 1024, down_sampling = False, spectral_normed = spectral_normed)
        
        # self.fully_connect_gan2 = nn.Linear(1024, 1)
        # torch.nn.init.normal_(self.fully_connect_gan2.weight.data, std=0.02)
        self.fully_connect_gan2 = SNLinear(1024, 1, spectral_normed = spectral_normed)
                
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        re1 = self.re1(x)
        re2 = self.re2(re1)
        re3 = self.re3(re2)
        re4 = self.re4(re3)
        re5 = self.re5(re4)
        re6 = self.re6(re5)
        re6 = self.relu(re6)
        re6 = torch.sum(re6,dim = (2,3))
        gan_logits = self.fully_connect_gan2(re6)
        return self.sigmoid(gan_logits), gan_logits



class SNGAN_G128(nn.Module):
    def __init__(self, z_size=128, channel=3, batch_normed=False):
        super(SNGAN_G128, self).__init__()

        self.z_size = z_size
        # self.spectral_normed = spectral_normed
        self.batch_normed = batch_normed
        
        self.fully_connect = nn.Linear(z_size, 4*4*1024)
        torch.nn.init.normal_(self.fully_connect.weight.data, std=0.02)
       
        self.re0 = Residual_G(1024, 1024, up_sampling = True, batch_normed=batch_normed)
        self.re1 = Residual_G(1024, 512, up_sampling = True, batch_normed=batch_normed)
        self.re2 = Residual_G(512, 256, up_sampling = True, batch_normed=batch_normed)
        self.re3 = Residual_G(256, 128, up_sampling = True, batch_normed=batch_normed)
        self.re4 = Residual_G(128, 64, up_sampling = True, batch_normed=batch_normed)
        
        self.bn = BatchNorm2d(64)
        self.conv_res4 = conv2d(64, channel, padding = 1, kernel_size = 3, stride = 1)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        d1 = self.fully_connect(x)
        d1 = d1.view(-1, 1024, 4, 4)

        
        d1 = self.re0(d1)
        d2 = self.re1(d1)
        d3 = self.re2(d2)
        d4 = self.re3(d3)
        d4 = self.re4(d4)
        if self.batch_normed:
            d4 = self.bn(d4)
        d5 = self.conv_res4(self.relu(d4))

        return self.tanh(d5)

    def sample_latent(self, num_samples):
        return torch.randn((num_samples, self.z_size))

