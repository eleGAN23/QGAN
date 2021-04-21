# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 17:09:38 2020

@author: Edoardo
"""
import torch.nn as nn

def QSSweights_init(m):

    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.r_weight.data, 0.0, 0.02)
        nn.init.normal_(m.i_weight.data, 0.0, 0.02)
        nn.init.normal_(m.j_weight.data, 0.0, 0.02)
        nn.init.normal_(m.k_weight.data, 0.0, 0.02)

    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.gamma.data, 1.0, 0.02)
        nn.init.constant_(m.beta.data, 0)
    
    
def SSweights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)