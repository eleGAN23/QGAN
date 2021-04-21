# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 11:32:43 2020

@author: Edoardo
"""


def GetModel(str_model,  z_size , BN=False, SN=False, ssup=False, needs_init=True):
    'Models: DCGAN_64, QDCGAN_64, SSGAN_32, QSSGAN_32, SSGAN_128, QSSGAN_128, QSSGAN_128_QSN'
    print('Model:', str_model)
    print()
    if str_model == 'DCGAN_64':
        from models.Q_DCGAN_64 import DCGAN_Discriminator, DCGAN_Generator, QDCGAN_Discriminator, QDCGAN_Generator
        return (DCGAN_Generator(batch_normed=BN, z_size = z_size, channel = 3, needs_init=needs_init), 
                    DCGAN_Discriminator(batch_normed=BN, channel = 3, needs_init=needs_init))

    elif str_model == 'QDCGAN_64':
        from models.Q_DCGAN_64 import DCGAN_Discriminator, DCGAN_Generator, QDCGAN_Discriminator, QDCGAN_Generator
        return (QDCGAN_Generator(batch_normed=BN, z_size = z_size, channel = 4),
                    QDCGAN_Discriminator(batch_normed=BN, channel = 4))
    
    elif str_model == 'SSGAN_32':
        from models.RotGAN_32 import RotGAN_G32, RotGAN_D32
        return (RotGAN_G32(batch_normed=BN, z_size = 128, channel = 3),
                    RotGAN_D32(spectral_normed = SN, ssup = ssup, channel = 3))
    
    elif str_model == 'QSSGAN_32':
        from models.QRotGAN_32 import QRotGAN_G32, QRotGAN_D32
        return (QRotGAN_G32(batch_normed=BN, z_size = 128, channel = 4),
                    QRotGAN_D32(spectral_normed = SN, ssup = ssup, channel = 4))
    
    elif str_model == 'QSSGAN_QSN_32':
        from models.QRotGAN_QSN_32 import QRotGAN_QSN_G32, QRotGAN_QSN_D32
        return (QRotGAN_QSN_G32(batch_normed=BN, z_size = 128, channel = 4),
                    QRotGAN_QSN_D32(spectral_normed = SN, ssup = ssup, channel = 4))
    
    elif str_model == 'SSGAN_128':
        from models.RotGAN_128 import RotGAN_G128, RotGAN_D128
        return (RotGAN_G128(batch_normed=BN, z_size = 128, channel = 3),
                    RotGAN_D128(spectral_normed = SN, ssup = ssup, channel = 3))
    
    elif str_model == 'QSSGAN_128':
        from models.QRotGAN_128 import QRotGAN_G128, QRotGAN_D128
        return (QRotGAN_G128(batch_normed=BN, z_size = 128, channel = 4),
                    QRotGAN_D128(spectral_normed = SN, ssup = ssup, channel = 4))
    
    elif str_model == 'QSSGAN2_128':
        from models.QRotGAN2_128 import QRotGAN2_G128, QRotGAN2_D128
        return (QRotGAN2_G128(batch_normed=BN, z_size = 128, channel = 4),
                    QRotGAN2_D128(spectral_normed = SN, ssup = ssup, channel = 4))
    
    
    elif str_model == 'QSSGAN_128_QSN':
        from models.QRotGAN_128_QSN import QRotGAN_G128_QSN, QRotGAN_D128_QSN
        return (QRotGAN_G128_QSN(batch_normed=BN, z_size = 128, channel = 4),
                    QRotGAN_D128_QSN(spectral_normed = SN, ssup = ssup, channel = 4))
    
    elif str_model == 'QSSGAN_128_FullQuat':
        from models.QRotGAN_128_FullQuat import QRotGAN_G128_FullQuat, QRotGAN_D128_FullQuat
        return (QRotGAN_G128_FullQuat(batch_normed=BN, z_size = 128, channel = 4),
                    QRotGAN_D128_FullQuat(spectral_normed = SN, ssup = ssup, channel = 4))
    
    
    elif str_model == 'CompGAN_32':
        from models.Compare_Torch_128_32 import CompGAN_D_32, CompGAN_G_32
        return (CompGAN_G_32(batch_normed=BN, z_size = 128, channel = 3),
                    CompGAN_D_32(spectral_normed = SN, ssup = ssup, channel = 3))
    
    
    elif str_model == 'CompGAN_128':
        from models.Compare_Torch_128_32 import CompGAN_D_128, CompGAN_G_128
        return (CompGAN_G_128(batch_normed=BN, z_size = 128, channel = 3),
                    CompGAN_D_128(spectral_normed = SN, ssup = ssup, channel = 3))
    
    
    
    else:
        raise ValueError ('Model not implemented, check allowed models (-help) \n \
             Models: DCGAN_64, QDCGAN_64, SSGAN_32, QSSGAN_32, SSGAN_128, QSSGAN_128')
        
        
