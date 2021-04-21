# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 10:24:26 2020

@author: Edoardo
"""
import sys
sys.path.append('../utils')
import argparse 
import torch
import torch.optim as optim
# from utils.Qdataloaders import get_CelebA_QDCGAN_dataloader, get_CelebA_DCGAN_dataloader
from utils.Qdataloaders import  CelebA_dataloader2, CelebA_colab_dataloader, CelebAHQ_dataloader, LSUN_dataloader
from utils.Qdataloaders import CIFAR10_dataloader

# from Qmodel import Generator, Discriminator, Simple_Discriminator, Simple_Generator
# from Qmodel import DCGAN_Generator, DCGAN_Discriminator
# from models.QRotGAN_32 import QRotGAN_G32, QRotGAN_D32
# from models.QRotGAN_128 import QRotGAN_G128, QRotGAN_D128
# from models.Q_DCGAN_64 import DCGAN_Discriminator, DCGAN_Generator, QDCGAN_Discriminator, QDCGAN_Generator
# from models.Test_QGAN import QGANLastConv_D, QGANLastConv_G

from Qtraining_new import Trainer
# from torch import nn
import random
import numpy as np
import os
from GetModel import GetModel
from utils.readFile import readFile
from multiprocessing import cpu_count

if __name__ == '__main__':
    parser = argparse.ArgumentParser()#fromfile_prefix_chars='@')
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--gpu_num', type=int, default=1)
    parser.add_argument('--colab', type=bool, default=False)
    parser.add_argument('--n_workers', default='max')
    
    parser.add_argument('--train_dir', type=str, default='./data/celebA_Train/Train', help="Folder containg training data. It must point to a folder with images in it.")
    
    parser.add_argument('--Dataset', type=str, default='CelebA_GAN', help='CelebA_GAN, CelebAHQ_GAN')
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--normalize', type=bool, default=False, help='map value of images from range [0,255] to range [-1,1]')
    
    parser.add_argument('--model', type=str, default='DCGAN_64', help='Models: DCGAN_64, QDCGAN_64, SSGAN_32, QSSGAN_32, SSGAN_128, QSSGAN_128')
    parser.add_argument('--ssup', type=bool, default=False, help='Old option, set it False')
    parser.add_argument('--noise_dim', type=int, default=128)
    parser.add_argument('--BN', type=bool, default=False, help='Apply Batch Normalization')
    parser.add_argument('--SN', type=bool, default=False, help='Apply Spectral Normalization')    
    
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    
    parser.add_argument('--loss', type=str, default='hinge', help='[hinge, classic, wasserstein]')
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--betas', default=(0.0, 0.9))
    
    parser.add_argument('--crit_iter', type=int, default=1, help='critic iteration') 
    parser.add_argument('--gp_weight', type=int, default=0, help='[1,10] for SSGAN, default=0')
    parser.add_argument('--weight_rot_D', type=float, default=0, help='1.0 for SSGAN, default=0')
    parser.add_argument('--weight_rot_G', type=float, default=0, help='0.2 for SSGAN, default=0')
    
    parser.add_argument('--print_every', type=int, default=50, help='Print Gen and Disc Loss every n iterations')
    parser.add_argument('--plot_images', type=bool, default=True, help='Plot images during training')
    parser.add_argument('--save_images', type=bool, default=True, help='Save images every epoch to track performance')
    parser.add_argument('--EpochCheckpoints', type=bool, default=True, help='Save model every epoch. If set to False the model will be saved only at the end')
    
    parser.add_argument('--save_FID', type=bool, default=True, help='Save images and compute FID score')
    parser.add_argument('--Test_FID_dir', type=str, default='./data/celeba/img_align_celeba/Test_FID_100/', help='Path to Folder with Test images for FID')

    parser.add_argument('--TextArgs', type=str, default='TrainingArguments.txt', help='Path to text with training settings')

    parse_list=readFile(parser.parse_args().TextArgs)
    
    opt = parser.parse_args(parse_list)
    
    use_cuda = opt.cuda
    gpu_num = opt.gpu_num
    loss = opt.loss
    critic_iterations = opt.crit_iter  # [1, 2]
    gp_weight = opt.gp_weight         # [1, 10]
    ssup = opt.ssup
    weight_rotation_loss_d = opt.weight_rot_D    #1.0
    weight_rotation_loss_g = opt.weight_rot_G   #0.2
    
    lr = opt.lr
    betas = opt.betas.replace(',', ' ').split()
    betas = (float(betas[0]), float(betas[1]))
    # print(betas)
    epochs = opt.epochs
    noise_dim = opt.noise_dim
    BN = opt.BN # Batch normalization
    # print(BN)
    SN = opt.SN # Spectral Normalization
    
    save_FID = opt.save_FID
    plot_images = opt.plot_images
    
    img_size = opt.image_size
    batch_size = opt.batch_size
    print_every = opt.print_every
    EpochCheckpoints = opt.EpochCheckpoints
    save_images = opt.save_images
    
    dataset = opt.Dataset
    colab = opt.colab
    n_workers = opt.n_workers
    
    if n_workers=='max':
        n_workers = cpu_count()
    
    
    manualSeed = 999
    #manualSeed = random.randint(1, 10000) # use if you want new results
    # print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    seed=manualSeed
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    
    model = opt.model
    
    train_dir = opt.train_dir
    normalize = opt.normalize
    # print(model, BN, SN, ssup)
    generator, discriminator = GetModel(str_model=model, z_size=noise_dim, BN=BN, SN = SN, ssup=ssup)
    
    G_params= sum(p.numel() for p in generator.parameters() if p.requires_grad)
    print('G parameters:', G_params)
    D_params= sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
    print('D parameters:', D_params)
    print('Total parameters:', G_params+D_params)
    print()
    
    if 'Q' in generator.__class__.__name__:
        quat_data = True
    else:
        quat_data= False
            
    if dataset == 'CelebA_GAN':
        if not colab:
            data_loader, _ , data_name = CelebA_dataloader2(root=train_dir, quat_data = quat_data, normalize=normalize, batch_size=batch_size, img_size=img_size, num_workers=n_workers)
            # print('Padded dataset loaded')
        else:
            data_loader, _ , data_name = CelebA_colab_dataloader(root=train_dir, quat_data = quat_data, normalize=normalize, batch_size=batch_size, img_size=img_size, num_workers=n_workers)
    
    elif dataset == 'CelebAHQ_GAN':
        data_loader, _ , data_name = CelebAHQ_dataloader(root=train_dir, quat_data = quat_data, normalize=normalize, batch_size=batch_size, img_size=img_size, num_workers=n_workers)
        
    elif dataset == 'LSUN_Bedroom':
        data_loader, _ , data_name = LSUN_dataloader(root=train_dir, quat_data = quat_data, normalize=normalize, batch_size=batch_size, img_size=img_size, num_workers=n_workers)

    elif dataset == 'CIFAR10':
        data_loader, _ , data_name = CIFAR10_dataloader(root=train_dir, quat_data = quat_data, normalize=normalize, batch_size=batch_size, img_size=img_size, num_workers=n_workers)

    # elif dataset == 'CelebA_128':
    #     data_loader, _ , data_name = get_CelebA_Rot_dataloader(root=opt.train_dir, batch_size=batch_size, img_size=img_size, num_workers=2)
    else:
        RuntimeError('Wrong dataset or not implemented')
    
    
    
    gen_img_path = './generated_images/'
    real_img_path = opt.Test_FID_dir
    FID_paths = [gen_img_path, real_img_path]
    
    
    checkpoint_folder = 'checkpoints/'
    if not os.path.isdir(checkpoint_folder):
        os.makedirs(checkpoint_folder)
    
    # Initialize optimizers
    G_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=betas)
    D_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=betas)
    
    '''Train model'''
    trainer = Trainer(generator, discriminator, G_optimizer, D_optimizer,
                      weight_rotation_loss_d = weight_rotation_loss_d, weight_rotation_loss_g = weight_rotation_loss_g,
                      use_cuda=use_cuda, gpu_num=gpu_num, print_every = print_every,
                      loss = loss,
                      gp_weight=gp_weight,
                      critic_iterations=critic_iterations,
                      save_FID = save_FID,
                      FIDPaths = [gen_img_path, real_img_path],
                      ssup=ssup,
                      checkpoint_folder = checkpoint_folder,
                      plot_images=plot_images,
                      save_images=save_images,
                      saveModelsPerEpoch=EpochCheckpoints,
                      normalize = normalize,
                      )
    
    
    trainer.train(data_loader, epochs)
