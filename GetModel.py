# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 11:32:43 2020

@author: Edoardo
"""


def GetModel(str_model,  z_size , BN=False, SN=False, ssup=False, needs_init=True):
    'Models: DCGAN_64, QDCGAN_64, SSGAN_32, QSSGAN_32, SSGAN_128, QSSGAN_128, QSSGAN_128_QSN'
    print('Model:', str_model)
    print()
    if str_model == 'SNGAN_32':
        from models.SNGAN_32 import SNGAN_G32, SNGAN_D32
        return (SNGAN_G32(batch_normed=BN, z_size = 128, channel = 3),
                    SNGAN_D32(spectral_normed = SN, ssup = ssup, channel = 3))
    
    elif str_model == 'QSNGAN_QSN_32':
        from models.QSNGAN_QSN_32 import QSNGAN_QSN_G32, QSNGAN_QSN_D32
        return (QSNGAN_QSN_G32(batch_normed=BN, z_size = 128, channel = 4),
                    QSNGAN_QSN_D32(spectral_normed = SN, ssup = ssup, channel = 4))
    
    elif str_model == 'SNGAN_128':
        from models.SNGAN_128 import SNGAN_G128, SNGAN_D128
        return (SNGAN_G128(batch_normed=BN, z_size = 128, channel = 3),
                    SNGAN_D128(spectral_normed = SN, ssup = ssup, channel = 3))
     
    elif str_model == 'QSNGAN_128_QSN':
        from models.QSNGAN_128_QSN import QSNGAN_G128_QSN, QSNGAN_D128_QSN
        return (QSNGAN_G128_QSN(batch_normed=BN, z_size = 128, channel = 4),
                    QSNGAN_D128_QSN(spectral_normed = SN, ssup = ssup, channel = 4))   
    
    
    else:
        raise ValueError ('Model not implemented, check allowed models (-help) \n \
             Models: DCGAN_64, QDCGAN_64, SSGAN_32, QSSGAN_32, SSGAN_128, QSSGAN_128')
        
        
