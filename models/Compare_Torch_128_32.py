# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 15:38:22 2021

@author: Edoardo
"""

import torch
import torch.nn as nn
import torch.nn.functional as F



# class Upsample(nn.Module): # old, da compare_gan
#     def __init__(self):
#         super(Upsample,self).__init__()    
        
#     def forward(self,input):
#         value = input.permute(0,2,3,1)
#         sh = [int(x) for x in value.shape]
#         dim = len(sh[1:-1])
        
#         out = (torch.reshape(value, [-1] + sh[-dim:]))
        
#         for i in range(dim, 0, -1):
        
#           out = torch.cat([out, torch.zeros(out.shape, device=(out.device))], i)
        
#         out_size = [-1] + [s * 2 for s in sh[1:-1]] + [sh[-1]]
#         out = torch.reshape(out, out_size)

#         return out.permute(0,3,1,2)


class Upsample(nn.Module): # old, da compare_gan
    def __init__(self):
        super(Upsample,self).__init__()    
        
    
    def _upsample(self, input):
        
        shape = input.shape
        up = F.upsample(input, scale_factor=2)
        pool, indices = F.max_pool2d_with_indices(up, kernel_size=2, stride=2, return_indices=True)
        
        unpool = F.max_unpool2d(pool, indices, kernel_size=2, stride=2, output_size=(shape[0], shape[1], shape[2]*2, shape[3]*2))
        
        return unpool
    
    def forward(self,input):
        return self._upsample(input)

# def conv2d(inputs, output_dim, k_h, k_w, d_h, d_w, stddev=0.02, name="conv2d",
#            use_sn=False, use_bias=True):
#   """Performs 2D convolution of the input."""
#   # with tf.variable_scope(name):
#     w = tf.get_variable(
#         "kernel", [k_h, k_w, inputs.shape[-1].value, output_dim],
#         initializer=weight_initializer(stddev=stddev))
#     if use_sn:
#       w = spectral_norm(w)
#     outputs = tf.nn.conv2d(inputs, w, strides=[1, d_h, d_w, 1], padding="SAME")
#     if use_bias:
#       bias = tf.get_variable(
#           "bias", [output_dim], initializer=tf.constant_initializer(0.0))
#       outputs += bias
#   return outputs


def spectral_norm(module, name='weight'):
    module = torch.nn.utils.spectral_norm(module, name=name)
    return module



class conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, padding, kernel_size = 4, stride = 2,
                spectral_normed = False):
        super(conv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = padding)
        torch.nn.init.normal_(self.conv.weight.data, std=0.02)
            
        if spectral_normed:
            self.conv = spectral_norm(self.conv)

    def forward(self, input):
        out = self.conv(input)
        return out


# class ResNetBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, scale, SN):
#         super(self,ResNetBlock).__init__()
        
        
        
        
        
        
        
        
    
class _get_conv(nn.Module):
    def __init__(self,in_channels, out_channels,  scale, SN, 
            kernel_size=(3, 3), strides=(1, 1), padding=0):
        super(_get_conv,self).__init__()

        self._spectral_norm = SN
        self.scale = scale
        if scale == "up":
          self.Up = Upsample()
        
        self.Conv = conv2d(in_channels, out_channels, padding = padding, kernel_size=kernel_size, stride=strides, spectral_normed=self._spectral_norm)

        if scale == "down":
          self.Down = nn.AvgPool2d(2, 2, padding = 1)
        
        
    def forward(self, x):
        if self.scale == "up":
            x = self.Up(x)
                
        x = self.Conv(x)
            
        if self.scale == "down":
            x = self.Down(x)
            
        return x
    



class ResNetBlock(nn.Module):
    def __init__(self,
               in_channels,
               out_channels,
               scale,
               is_gen_block,
               batch_norm=False,
               spectral_norm=False,
               ):
        super(ResNetBlock, self).__init__()
        """Constructs a new ResNet block.
        Args:
          name: Scope name for the resent block.
          in_channels: Integer, the input channel size.
          out_channels: Integer, the output channel size.
          scale: Whether or not to scale up or down, choose from "up", "down" or
            "none".
          is_gen_block: Boolean, deciding whether this is a generator or
            discriminator block.
          layer_norm: Apply layer norm before both convolutions.
          spectral_norm: Use spectral normalization for all weights.
          batch_norm: Function for batch normalization.
        """
        assert scale in ["up", "down", "none"]
        
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._scale = scale
        # In SN paper, if they upscale in generator they do this in the first conv.
        # For discriminator downsampling happens after second conv.
        self._scale1 = scale if is_gen_block else "none"
        self._scale2 = "none" if is_gen_block else scale
        # self._layer_norm = layer_norm
        self._spectral_norm = spectral_norm
        self.batch_norm = batch_norm
    
        self.shortcut = _get_conv(self._in_channels, self._out_channels, kernel_size=1, padding=0, scale = self._scale, SN=self._spectral_norm)
        
        
        if self.batch_norm:
            self.output = nn.Sequential(nn.BatchNorm2d(self._in_channels),
                                    nn.ReLU(),
                                    _get_conv(self._in_channels, self._out_channels, padding=1, scale = self._scale1, SN=self._spectral_norm),
                                    nn.BatchNorm2d(self._out_channels),
                                    nn.ReLU(),
                                    _get_conv(self._out_channels, self._out_channels, padding=1, scale = self._scale2, SN=self._spectral_norm)) 
        
        else:    
        #     self.bn1 = nn.BatchNorm2d(self._in_channels)
        #     self.bn2 = nn.BatchNorm2d(self._out_channels)
            
        # self.relu = nn.ReLU()
        # self.conv1 = _get_conv(self._in_channels, self._out_channels, padding=1, scale = self._scale1, SN=self._spectral_norm)
        
        # self.conv2 = _get_conv(self._out_channels, self._out_channels, padding=1, scale = self._scale2, SN=self._spectral_norm)
    
            self.output = nn.Sequential(
                                    nn.ReLU(),
                                    _get_conv(self._in_channels, self._out_channels, padding=1, scale = self._scale1, SN=self._spectral_norm),
                                    
                                    nn.ReLU(),
                                    _get_conv(self._out_channels, self._out_channels, padding=1, scale = self._scale2, SN=self._spectral_norm)) 
    
    # sequenza
    # self.BN1 = nn.BatchNorm2d()
    # self.relu = nn.ReLU()
    # self.conv1 = _get_conv(self._in_channels, self._out_channels, self._scale1, SN=self._spectral_norm)
    # self.BN2 = nn.BatchNorm2d()
    # self.relu = nn.ReLU()(output)
    # self.conv2 = _get_conv(self._out_channels, self._out_channels, self._scale2, SN=self._spectral_norm)


    def forward(self, x):
        # print(self.output(x).shape)
        # print(self.shortcut(x).shape)
        return self.output(x) + self.shortcut(x)
    
    
    
    
class _get_linear(nn.Module):
    def __init__(self,in_features, out_features, use_bias=True, spectral_norm=False):
        super(_get_linear,self).__init__()

        self._spectral_norm = spectral_norm

        self.linear = nn.Linear(in_features, out_features, bias=use_bias)
        
        if spectral_norm:
            self.linear = torch.nn.utils.spectral_norm(self.linear)
            
        torch.nn.init.normal_(self.linear.weight.data, std=0.02)
        
        
    def forward(self, x):
        
        return self.linear(x)    
    
    
    
    
class CompGAN_G_128(nn.Module):
    def __init__(self, z_size=128, channel=3, batch_normed=True):
        super(CompGAN_G_128, self).__init__()
        
        self.z_size = z_size
        self.batch_normed = batch_normed
        self.linear = _get_linear(z_size, 4*4*1024, spectral_norm=False)
        # reshape
        
        self.res1 = ResNetBlock(1024, 1024, scale='up', is_gen_block=True)
        self.res2 = ResNetBlock(1024, 512, scale='up', is_gen_block=True)
        self.res3 = ResNetBlock(512, 256, scale='up', is_gen_block=True)
        self.res4 = ResNetBlock(256, 128, scale='up', is_gen_block=True)
        self.res5 = ResNetBlock(128, 64, scale='up', is_gen_block=True)
    
        self.BN = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        
        self.conv = conv2d(64, channel, padding=1, kernel_size=3, stride=1)

        
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        
        h = self.linear(x)
        h = h.view(-1, 1024, 4 ,4 )
        # print(h)
        h = self.res1(h)
        h = self.res2(h)
        h = self.res3(h)
        h = self.res4(h)
        h = self.res5(h)
        h = self.BN(h)
        h = self.relu(h)
        h = self.conv(h)
        h = self.tanh(h) # qui usano la sigmoid in compare_gan
        # print(h.shape)
        
        return h
    
    def sample_latent(self, num_samples):
        return torch.randn((num_samples, self.z_size))


    

    


class CompGAN_D_128(nn.Module):
    def __init__(self, channel=3, ssup=False, spectral_normed=True):
        super(CompGAN_D_128, self).__init__()

        self.spectral_normed = spectral_normed
        self.ssup = ssup
        # self.linear = nn.Linear(z_size, 4*4*1024)
        # reshape
        
        self.res1 = ResNetBlock(channel, 64, scale='down', is_gen_block=False, spectral_norm = self.spectral_normed)
        self.res2 = ResNetBlock(64, 128, scale='down', is_gen_block=False, spectral_norm = self.spectral_normed)
        self.res3 = ResNetBlock(128, 256, scale='down', is_gen_block=False, spectral_norm = self.spectral_normed)
        self.res4 = ResNetBlock(256, 512, scale='down', is_gen_block=False, spectral_norm = self.spectral_normed)
        self.res5 = ResNetBlock(512, 1024, scale='down', is_gen_block=False, spectral_norm = self.spectral_normed)
        self.res6 = ResNetBlock(1024, 1024, scale='none', is_gen_block=False, spectral_norm = self.spectral_normed)

    
        # self.BN = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        
        # self.conv = conv2d(64, channel, padding=1, kernel_size=3, stride=1)
        self.linear = _get_linear(1024, 1, spectral_norm=spectral_norm)
        
        if self.ssup:
            self.ssLinear = _get_linear(1024, 4, spectral_norm=spectral_norm)
        
        # self.tanh = nn.Tanh()
        
    def forward(self, x):
        

        # print(h)
        h = self.res1(x)
        h = self.res2(h)
        h = self.res3(h)
        h = self.res4(h)
        h = self.res5(h)
        h = self.res6(h)
        h = self.relu(h)
        # h = torch.sum(h,dim = (2,3))
        h = torch.mean(h,dim = (2,3))
        out = self.linear(h)
        
        # return out
    
        if self.ssup:
            rot_logits = self.ssLinear(h)
        

        if self.ssup:
            return torch.sigmoid(out), out, rot_logits, torch.softmax(rot_logits, dim=1)
        else:
            return torch.sigmoid(out), out
 
    
 
# g = CompGAN_G_128()   
# # print(g)
# t = torch.randn(1,128)
# g(t)
    
# t = torch.randn(1,3,128,128)    
    
# d = CompGAN_D_128()    
    
    
# print(d(t)    ) 
    
 
    
 
 
# ----------------------- 32x32 ( CIFAR10 ) --------------------------------------#    
 



class CompGAN_G_32(nn.Module):
    def __init__(self, z_size=128, channel=3, batch_normed=True):
        super(CompGAN_G_32, self).__init__()
        
        self.z_size = z_size
        self.batch_normed = batch_normed
        self.linear = _get_linear(z_size, 4*4*256, spectral_norm=False)
        # reshape
        
        self.res1 = ResNetBlock(256, 256, scale='up', is_gen_block=True)
        self.res2 = ResNetBlock(256, 256, scale='up', is_gen_block=True)
        self.res3 = ResNetBlock(256, 256, scale='up', is_gen_block=True)

    
        self.BN = nn.BatchNorm2d(256)
        
        self.conv = conv2d(256, channel, padding=1, kernel_size=3, stride=1)

        
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        
        h = self.linear(x)
        h = h.view(-1, 256, 4 ,4 )
        # print(h)
        h = self.res1(h)
        h = self.res2(h)
        h = self.res3(h)
        h = self.BN(h)
        h = torch.relu(h)
        h = self.conv(h)
        h = self.tanh(h) # qui usano la sigmoid in compare_gan
        # print(h.shape)
        
        return h
    
    def sample_latent(self, num_samples):
        return torch.randn((num_samples, self.z_size))
    
# g = CompGAN_G_32()   
# # print(g)
# t = torch.randn(10,128)
# print(g(t).shape)
    
# t = torch.randn(1,3,128,128)    
    
# d = CompGAN_D_128()    
    
    
# print(d(t)    ) 

    

    


class CompGAN_D_32(nn.Module):
    def __init__(self, channel=3, ssup=False, spectral_normed=True):
        super(CompGAN_D_32, self).__init__()

        self.spectral_normed = spectral_normed
        self.ssup = ssup
        # self.linear = nn.Linear(z_size, 4*4*1024)
        # reshape
        
        self.res1 = ResNetBlock(channel, 128, scale='down', is_gen_block=False, spectral_norm = self.spectral_normed)
        self.res2 = ResNetBlock(128, 128, scale='down', is_gen_block=False, spectral_norm = self.spectral_normed)
        self.res3 = ResNetBlock(128, 128, scale='none', is_gen_block=False, spectral_norm = self.spectral_normed)
        self.res4 = ResNetBlock(128, 128, scale='none', is_gen_block=False, spectral_norm = self.spectral_normed)

    
        # self.BN = nn.BatchNorm2d(64)
        
        # self.conv = conv2d(64, channel, padding=1, kernel_size=3, stride=1)
        self.linear = _get_linear(128, 1, spectral_norm=spectral_norm)
        
        if self.ssup:
            self.ssLinear = _get_linear(128, 4, spectral_norm=spectral_norm)
        
        # self.tanh = nn.Tanh()
        
    def forward(self, x):
        

        # print(h)
        h = self.res1(x)
        h = self.res2(h)
        h = self.res3(h)
        h = self.res4(h)
        h = torch.relu(h)
        # h = torch.sum(h,dim = (2,3))
        h = torch.mean(h,dim = (2,3))
        # print(h.shape)
        out = self.linear(h)
        
        # return out
    
        if self.ssup:
            rot_logits = self.ssLinear(h)
        

        if self.ssup:
            return torch.sigmoid(out), out, rot_logits, torch.softmax(rot_logits, dim=1)
        else:
            return torch.sigmoid(out), out    
    
    
    
# t = torch.randn(1,3,32,32)    
    
# d = CompGAN_D_32()    
    
    
# print(d(t)    )         
        