# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 14:10:20 2020

@author: Edoardo
"""


import sys
sys.path.append('../utils')

from utils.Ops import conv2d, Residual_G, Residual_D, spectral_norm, First_Residual_D, SNLinear
import torch
import torch.nn as nn
from torch.nn import BatchNorm2d



class RotGAN_D32(nn.Module):
    def __init__(self, ssup, channel=3, spectral_normed=False):
        super(RotGAN_D32, self).__init__()

        self.ssup = ssup
        self.spectral_normed = spectral_normed

        
        self.re1 = First_Residual_D(channel, 128, spectral_normed = spectral_normed)
        self.re2 = Residual_D(128, 128, down_sampling = True, spectral_normed = spectral_normed)
        self.re3 = Residual_D(128, 128, down_sampling = False, spectral_normed = spectral_normed)
        self.re4 = Residual_D(128, 128, down_sampling = False, spectral_normed = spectral_normed)
        
        # self.fully_connect_gan2 = nn.Linear(128, 1, bias=False)
        # # nn.init.xavier_uniform_(self.fully_connect_gan2.weight.data, 1.)
        # torch.nn.init.normal_(self.fully_connect_gan2.weight.data, std=0.02)
        self.fully_connect_gan2 = SNLinear(128, 1, bias=False, spectral_normed = spectral_normed)
            
        # self.fully_connect_rot2 = nn.Linear(128, 4, bias=False)
        # # nn.init.xavier_uniform_(self.fully_connect_rot2.weight.data, 1.)
        # torch.nn.init.normal_(self.fully_connect_rot2.weight.data, std=0.02)
        self.fully_connect_rot2 = SNLinear(128, 4, bias=False, spectral_normed = spectral_normed)
        
        # if spectral_normed:
        #     self.fully_connect_gan2 = spectral_norm(self.fully_connect_gan2)
        #     self.fully_connect_rot2 = spectral_norm(self.fully_connect_rot2)
        
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()

        

    def forward(self, x):

        re1 = self.re1(x)
        re2 = self.re2(re1)
        re3 = self.re3(re2)
        re4 = self.re4(re3)
        re4 = self.relu(re4)
        re4 = torch.sum(re4,dim = (2,3))
        gan_logits = self.fully_connect_gan2(re4)
        
        if self.ssup:
            rot_logits = self.fully_connect_rot2(re4)
            rot_prob = self.softmax(rot_logits)

        if self.ssup:
            return self.sigmoid(gan_logits), gan_logits, rot_logits, rot_prob
        else:
            return self.sigmoid(gan_logits), gan_logits



class RotGAN_G32(nn.Module):
    def __init__(self, z_size=128, channel=3, batch_normed=False):
        super(RotGAN_G32, self).__init__()
        s = 4
        self.s = s
        self.z_size = z_size
        self.batch_normed = batch_normed

        self.fully_connect = nn.Linear(z_size, s*s*256)
        # torch.nn.init.xavier_uniform_(self.fully_connect.weight.data, gain=1.0)
        torch.nn.init.normal_(self.fully_connect.weight.data, std=0.02)
        
        self.conv_res4 = conv2d(256,channel, padding = 1, kernel_size = 3, stride = 1)
        # torch.nn.init.xavier_uniform_(self.conv_res4.conv.weight.data, gain=1.0)
        # torch.nn.init.normal_(self.conv_res4.weight.data, std=0.02)
        
        self.re1 = Residual_G(256, 256, up_sampling = True, batch_normed=batch_normed)
        self.re2 = Residual_G(256, 256, up_sampling = True, batch_normed=batch_normed)
        self.re3 = Residual_G(256, 256, up_sampling = True, batch_normed=batch_normed)
        self.bn = BatchNorm2d(256)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
        
        
    def forward(self, x):
        d1 = self.fully_connect(x)
        d1 = d1.view(-1, 256, 4, 4)

        d2 = self.re1(d1)
        d3 = self.re2(d2)
        d4 = self.re3(d3)
        if self.batch_normed:
            d4 = self.bn(d4)
        d4 = self.relu(d4)
        d5 = self.conv_res4(d4)

        return self.tanh(d5)

    def sample_latent(self, num_samples):
        return torch.randn((num_samples, self.z_size))

