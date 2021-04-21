# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 23:00:11 2020

@author: Edoardo
"""

import sys
sys.path.append('../utils')
import torch
import torch.nn as nn
# from utils.QBatchNorm import QuaternionBatchNormalization as QBatchNorm
# from utils.bn_Pytorch import QuaternionBatchNormalization as QBatchNorm
# from utils.QEdoBN import QuaternionBatchNormalization as QBatchNorm
# from utils.QPYbn_new import QuaternionBatchNorm2d as QBatchNorm
# from utils.QPYbn_new import NQuaternionBatchNorm2d as QBatchNorm
from utils.QBN_Vecchi2 import QuaternionBatchNorm2d as QBatchNorm

from utils.quaternion_layers import QuaternionConv, QuaternionTransposeConv




def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        
        
def Qweights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.r_weight.data, 0.0, 0.02)
        nn.init.normal_(m.i_weight.data, 0.0, 0.02)
        nn.init.normal_(m.j_weight.data, 0.0, 0.02)
        nn.init.normal_(m.k_weight.data, 0.0, 0.02)

    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.gamma.data, 1.0, 0.02)
        nn.init.constant_(m.beta.data, 0)


# DCGAN Discriminator (output )
class DCGAN_Discriminator(nn.Module):
    def __init__(self, channel=3,
                 featD=64,
                 Fout_D=1,
                 batch_normed=False,
                 needs_init = True):
        
        super(DCGAN_Discriminator, self).__init__()
    
            
        self.conv1 = nn.Conv2d(channel, featD, 4, 2, 1, bias=False)
        # input is (nc) x 64 x 64
        
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        # state size. (ndf) x 32 x 32
        self.conv2 = nn.Conv2d(featD, featD * 2, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(featD * 2)
        # nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf*2) x 16 x 16
        self.conv3 = nn.Conv2d(featD * 2, featD * 4, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(featD * 4)
        # nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf*4) x 8 x 8
        self.conv4 = nn.Conv2d(featD * 4, featD * 8, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(featD * 8)
        # nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf*8) x 4 x 4
        self.conv5 = nn.Conv2d(featD * 8, Fout_D, 4, 1, 0, bias=False)
        self.sigmoid = nn.Sigmoid()
        
        self.batch_normed = batch_normed
        self.needs_init = needs_init

    def forward(self, input):
        l1 = self.lrelu(self.conv1(input))
        if self.batch_normed:
            # print(self.batch_normed)
            l2 = self.lrelu(self.bn2(self.conv2(l1)))
            l3 = self.lrelu(self.bn3(self.conv3(l2)))
            l4 = self.lrelu(self.bn4(self.conv4(l3)))
        else:
            l2 = self.lrelu(self.conv2(l1))
            l3 = self.lrelu(self.conv3(l2))
            l4 = self.lrelu(self.conv4(l3))
            
        output = self.conv5(l4)
        
        return self.sigmoid(output), output
    




# DCGAN Generator (output 3x64x64)
class DCGAN_Generator(nn.Module):
    def __init__(self, z_size = 128,
                         ngf = 64,
                         channel = 3,
                         batch_normed=False,
                         needs_init = True):
        super(DCGAN_Generator, self).__init__()
        
        self.z_size = z_size
            
        # input is Z, going into a convolution
        self.deconv1 = nn.ConvTranspose2d( z_size, ngf * 8, 4, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(ngf * 8)
        self.relu = nn.ReLU(True)
        # state size. (ngf*8) x 4 x 4
        self.deconv2 = nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ngf * 4)

        # state size. (ngf*4) x 8 x 8
        self.deconv3 = nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(ngf * 2)

        # state size. (ngf*2) x 16 x 16
        self.deconv4 = nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(ngf)

        # state size. (ngf) x 32 x 32
        self.deconv5 = nn.ConvTranspose2d( ngf, channel, 4, 2, 1, bias=False)
        
        # state size. (nc) x 64 x 64
        
        
        self.tan = nn.Tanh()
        
        self.batch_normed = batch_normed
        self.needs_init = needs_init

    
    def forward(self, input):
        
        if self.batch_normed:
            l1 = self.relu(self.bn1(self.deconv1(input)))
            l2 = self.relu(self.bn2(self.deconv2(l1)))
            l3 = self.relu(self.bn3(self.deconv3(l2)))
            l4 = self.relu(self.bn4(self.deconv4(l3)))
            
        else:
            l1 = self.relu(self.deconv1(input))
            l2 = self.relu(self.deconv2(l1))
            l3 = self.relu(self.deconv3(l2))
            l4 = self.relu(self.deconv4(l3))
        
        output = self.tan(self.deconv5(l4))
            
            
        return output

    def sample_latent(self, num_samples):
        return torch.randn(num_samples, self.z_size, 1, 1)

















# QDCGAN Discriminator (output )
class QDCGAN_Discriminator(nn.Module):
    def __init__(self, channel=4,
                 featD=64,
                 Fout_D=1,
                 batch_normed=False,
                 needs_init = False):
        
        super(QDCGAN_Discriminator, self).__init__()
      
            
        self.conv1 = QuaternionConv(channel, featD, kernel_size=4, stride=2, padding=1, bias=False)
        # input is (nc) x 64 x 64
        
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        # state size. (ndf) x 32 x 32
        self.conv2 = QuaternionConv(featD, featD * 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = QBatchNorm(featD * 2)
        # nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf*2) x 16 x 16
        self.conv3 = QuaternionConv(featD * 2, featD * 4, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = QBatchNorm(featD * 4)
        # nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf*4) x 8 x 8
        self.conv4 = QuaternionConv(featD * 4, featD * 8, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn4 = QBatchNorm(featD * 8)
        # nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf*8) x 4 x 4
        self.conv5 = QuaternionConv(featD * 8, 4, kernel_size=4, stride=1, padding=0, bias=False)
        self.dense = nn.Linear(4,Fout_D)
        self.sigmoid = nn.Sigmoid()
        self.flat = nn.Flatten()
        
        self.batch_normed = batch_normed
        

    def forward(self, input):
        # print(input.shape)
        l1 = self.lrelu(self.conv1(input))
        if self.batch_normed:
            l2 = self.lrelu(self.bn2(self.conv2(l1)))
            l3 = self.lrelu(self.bn3(self.conv3(l2)))
            l4 = self.lrelu(self.bn4(self.conv4(l3)))
        else:
            l2 = self.lrelu(self.conv2(l1))
            l3 = self.lrelu(self.conv3(l2))
            l4 = self.lrelu(self.conv4(l3))
        # print(l4.shape)    
        output = self.conv5(l4)
        output = self.flat(output)
        output = self.dense(output)
        
        return self.sigmoid(output), output









# QDCGAN Generator (output 3x64x64)
class QDCGAN_Generator(nn.Module):
    def __init__(self, z_size = 128,
                         ngf = 64,
                         channel = 4,
                         batch_normed=False,
                         needs_init = False):
        super(QDCGAN_Generator, self).__init__()
        

            
        # input is Z, going into a convolution
        self.deconv1 = QuaternionTransposeConv( z_size, ngf * 8, kernel_size=4, stride=1, padding=0, bias=False)
        self.bn1 = QBatchNorm(ngf * 8)
        self.relu = nn.ReLU(True)
        # state size. (ngf*8) x 4 x 4
        self.deconv2 = QuaternionTransposeConv(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = QBatchNorm(ngf * 4)

        # state size. (ngf*4) x 8 x 8
        self.deconv3 = QuaternionTransposeConv( ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = QBatchNorm(ngf * 2)

        # state size. (ngf*2) x 16 x 16
        self.deconv4 = QuaternionTransposeConv( ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn4 = QBatchNorm(ngf)

        # state size. (ngf) x 32 x 32
        self.deconv5 = QuaternionTransposeConv( ngf, channel, kernel_size=4, stride=2, padding=1, bias=False)
        
        # state size. (nc) x 64 x 64
        
        
        self.tan = nn.Tanh()
        
        self.batch_normed = batch_normed
        self.z_size = z_size

    
    def forward(self, input):
        if self.batch_normed:
            l1 = self.relu(self.bn1(self.deconv1(input)))
            l2 = self.relu(self.bn2(self.deconv2(l1)))
            l3 = self.relu(self.bn3(self.deconv3(l2)))
            l4 = self.relu(self.bn4(self.deconv4(l3)))
            
        else:
            l1 = self.relu(self.deconv1(input))
            l2 = self.relu(self.deconv2(l1))
            l3 = self.relu(self.deconv3(l2))
            l4 = self.relu(self.deconv4(l3))
        
        output = self.tan(self.deconv5(l4))
            
            
        return output

    def sample_latent(self, num_samples):
        return torch.randn((num_samples, self.z_size,1 ,1))
