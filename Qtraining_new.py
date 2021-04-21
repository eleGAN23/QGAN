
import numpy as np
import torch
import torch.nn as nn

from torch.autograd import Variable
from torch.autograd import grad as torch_grad
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
matplotlib.interactive(True)    
import time

import sys
sys.path.append('../utils')
import os

from utils.pytorch_fid.fid_score import FID
from datetime import datetime
import json
date = str(datetime.now()).replace(' ', '_')[:-7].replace(':', '-')

from models.Q_DCGAN_64 import weights_init, Qweights_init
# from utils.SSweights_Init import SSweights_init, QSSweights_init

import torchvision.utils as vutils

class Pad():
  def __init__(self):
    return
      
  def __call__(self, tensor):
    self.tensor = tensor
    channels = tensor.shape[1] # num channels in batch
    
    if channels == 3: 
      npad  = ((0,0), (1, 0), (0, 0), (0, 0))
      # self.tensor = np.pad(self.tensor, pad_width=npad, mode='constant', constant_values=0)
    elif channels == 1:
      self.tensor = torch.cat((self.tensor.data, self.tensor.data, self.tensor.data), dim=1)
      npad  = ((0,0), (1, 0), (0, 0), (0, 0))

    self.tensor = np.pad(self.tensor, pad_width=npad, mode='constant', constant_values=0)


    return torch.Tensor(self.tensor)


class Trainer():
    def __init__(self, generator, discriminator, gen_optimizer, dis_optimizer,
                 weight_rotation_loss_d, weight_rotation_loss_g, loss='hinge', gp_weight=10, critic_iterations=2, print_every=50,
                 use_cuda=True,
                 gpu_num=1,
                 save_FID=False,
                 FIDPaths = ['generated_images','real_images'],
                 ssup=False,
                 checkpoint_folder='checkpoints',
                 FIDevery = 500,
                 FIDImages = 100,
                 plot_images=False,
                 save_images=False,
                 saveModelsPerEpoch=True,
                 normalize=True):
        
        
        self.G = generator
        self.G_opt = gen_optimizer
        self.D = discriminator
        self.D_opt = dis_optimizer
        self.losses = {'LossG': [], 'LossD': [], 'GP': [], 'RotationG': [], 'RotationD': []}
        self.num_steps = 0
        self.use_cuda = use_cuda
        self.gpu_num = gpu_num
        self.gp_weight = gp_weight
        self.critic_iterations = critic_iterations
        self.print_every = print_every
        self.weight_rotation_loss_d = weight_rotation_loss_d
        self.weight_rotation_loss_g = weight_rotation_loss_g

#         if self.use_cuda:
        device = torch.device('cuda:%i' %gpu_num if torch.cuda.is_available() and self.use_cuda else 'cpu')
        self.device = device
        self.G.to(device)
        self.D.to(device)
        
        self.selected_loss = loss
        if self.selected_loss == 'classic':
            self.BCE_loss = nn.BCELoss()

        
        self.save_FID = save_FID
        # self.FID = FID()
        self.FIDPaths = FIDPaths
        self.FIDImages = FIDImages
        self.saveModelsPerEpoch = saveModelsPerEpoch
        self.save_images = save_images
        self.plot_images = plot_images # plot images during training
        self.normalize = normalize
        
        self.tracked_info = {'Epochs': 0, 'Iterations': 0, 'LossG': [], 'LossD': [], 'GP': [], 'FID': [], 'EpochFID': [], 'RotationG': [], 'RotationD': []}
        self.checkpoint_folder = checkpoint_folder
        self.ssup = ssup
        
        
        self.QNet = ('Q' in self.G.__class__.__name__ ) # check if network is Quaternionic
        
        # init weight of DCGAN and QDCGAN
        if hasattr(self.G, 'needs_init') and hasattr(self.D, 'needs_init'):
            if self.G.needs_init==True and self.D.needs_init==True:
                print('DCGAN/QDCGAN Weights init =', self.G.needs_init)
                if not self.QNet:
                    self.G.apply(weights_init)
                    # print(list(self.G.children()))
                    self.D.apply(weights_init)
                else:
                    self.G.apply(Qweights_init)
                    # print(list(self.G.children()))
                    self.D.apply(Qweights_init)
                   
        
        
      
        
        # info about Gen to put in folder's name or file name        
        self.Generator_info = self.G.__class__.__name__ + '_BN-{}_SN-{}_SSUP-{}'.format(
            self.G.batch_normed, hasattr(self.D, 'spectral_normed') and self.D.spectral_normed, self.ssup)
        
        # update generated images fid path
        self.FIDPaths[0] = str(self.FIDPaths[0]) + str(self.Generator_info)
        print('\nGenerated images saved in', self.FIDPaths[0])
       
        dir_gen = self.FIDPaths[0] #os.path.abspath(self.FIDPaths[0])
        # create folder for generated images
        if not os.path.isdir(dir_gen):
            os.makedirs(dir_gen)
            
            
        print('\nQuaternion Model = {}\nGenerator, Discriminator = {} and {}\nloss = {}\nSelf-Supervised = {}\nBatch Normalization = {}\
               \nSpectral Normalization = {}\n'.format(self.QNet,
              self.G.__class__.__name__, self.D.__class__.__name__, self.selected_loss, self.ssup, self.G.batch_normed,
                  hasattr(self.D, 'spectral_normed') and self.D.spectral_normed ==True))
        
        time.sleep(5)
            

    def _critic_train_iteration(self, data, generated_data, batch_size):
        """ Compute Discriminator Loss and Optimize """
        # Calculate probabilities on real and generated data
        # data divided in upright and rotated    
        # take first batch corresponding to the total images if the network is not self-supervised
        # or corresponding to the upright images if the network is self-supervised
        self.D.zero_grad()            

            
        else:
            if self.selected_loss != 'classic':
                all_data = torch.cat([data, generated_data], dim=0)
    
                sigmoid, logits = self.D(all_data)
                
                d_real_sigmoid, g_fake_sigmoid = torch.chunk(sigmoid, 2)
                d_real_logits, g_fake_logits = torch.chunk(logits, 2)
            
#             print("[D real: %f][D fake: %f]" %(torch.mean(d_real_sigmoid), torch.mean(g_fake_sigmoid)))
                
            
                


        if self.gp_weight > 0:
            data_up = data[0:batch_size]
            generated_data_up = generated_data[0:batch_size]
            
            # Get gradient penalty
            gradient_penalty = self._gradient_penalty(data_up.data, generated_data_up.data)
            self.losses['GP'].append(gradient_penalty.item())
           
           
        # Create D loss and optimize
        if self.selected_loss=='wasserstein':
            d_loss = torch.mean(g_fake_logits) - torch.mean(d_real_logits) 
            
        elif self.selected_loss== 'hinge':
            # print('d_real_pro_logits shape', d_real_pro_logits.shape)
            d_loss = torch.mean(nn.ReLU()(1.0 - d_real_logits.view(-1))) + torch.mean(nn.ReLU()(1.0 + g_fake_logits.view(-1)))
            
        elif self.selected_loss== 'classic':
            real_sigmoid, _ = self.D(data)
            errD = self.BCE_loss(real_sigmoid.view(-1), self.label_one)
            errD.backward()
            fake_sigmoid, _ = self.D(generated_data)
            errG = self.BCE_loss(fake_sigmoid.view(-1), self.label_zero)
            errG.backward()
            d_loss = errD + errG
        
        
        
        if self.gp_weight > 0 :
            if self.selected_loss!= 'classic':
                d_loss += gradient_penalty
            else:
                d_loss == gradient_penalty
        
        
        if self.selected_loss != 'classic':
            d_loss.backward()#retain_graph=True)
            
        # Optimize
        self.D_opt.step()

        # Record loss
        self.losses['LossD'].append(d_loss.item())


    def _generator_train_iteration(self, generated_data, batch_size):
        """ Compute Generator Loss and Optimize """
        # print('gen data size', generated_data.shape)
        self.G.zero_grad()
        
        g_fake_sigmoid, g_fake_logits = self.D(generated_data)
            
            
        
        if self.selected_loss in ['wasserstein', 'hinge']:
            g_loss =  -torch.mean(g_fake_logits.view(-1))
            
        elif self.selected_loss == 'classic':
            # self.label.fill_(1.)
            # print(g_fake_sigmoid.view(-1).shape)
            g_loss = self.BCE_loss(g_fake_sigmoid.view(-1), self.label_one)
                
        g_loss.backward()
        self.G_opt.step()

        # Record loss
        self.losses['LossG'].append(g_loss.item())


    def _gradient_penalty(self, real_data, generated_data):
        ''' Compute gradient penalty '''

        # Compute interpolation
        alpha = torch.rand(real_data.size(0), 1, 1, 1).to(self.device)    
        interpolated = Variable( alpha * real_data + (1 - alpha) * generated_data, requires_grad=True).to(self.device)

        # Compute probability of interpolated examples
        _, logit_interpolated = self.D(interpolated)
            
        out_interpolated_batch = logit_interpolated.size()
        
        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(outputs=logit_interpolated, inputs=interpolated,
                    grad_outputs=torch.ones(out_interpolated_batch).to(self.device) if self.use_cuda else torch.ones(
                     out_interpolated_batch),
                    create_graph=True, retain_graph=True, only_inputs=True)[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(gradients.size(0), -1)
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        gradient_penalty = torch.mean((gradients_norm - 1) ** 2)


        # Return gradient penalty
        return self.gp_weight * gradient_penalty


    def _train_epoch(self, data_loader):
        self.G.train()
        # start_time = time.time()
        for i, data in enumerate(data_loader):
            
            # Get generated data
            data = data[0].to(self.device)
#             print(data.size())
            
            # r= vutils.make_grid(data[0].permute(1,2,0).cpu().data, normalize=True, range=(-1,1))
            # plt.imshow(r)
            # plt.show()
            # plt.pause(0.5)
            batch_size = data.size(0)
            generated_data = self.sample_generator(batch_size)
                        
            # Prepare labels for classic loss
            if self.selected_loss == 'classic':
                self.label_one = torch.full((batch_size,), 1, dtype=torch.float, device=self.device)
                self.label_zero = torch.full((batch_size,), 0, dtype=torch.float, device=self.device)
            self.num_steps += 1

            
            # Update Discriminator (excluding rotated generated images)
            self._critic_train_iteration(data, generated_data.detach(), batch_size)
            
            # Only update generator every |critic_iterations| iterations
            if self.num_steps % self.critic_iterations == 0:
                self._generator_train_iteration(generated_data, batch_size)
                
            # Print Loss informations and plot generated images
            if self.num_steps % self.print_every == 0:
                # print('{} minutes'.format((time.time() - start_time)/60))
                
                print()
                print("Iteration {}".format(self.num_steps))
                print("Total Loss D: {}".format(self.losses['LossD'][-1]))
                    
                if len(self.losses['LossG']) !=0:
                    print("Total Loss G: {}".format(self.losses['LossG'][-1]))
                if self.gp_weight !=0:
                    print("GP: {}".format(self.losses['GP'][-1]))
                
                    # print("Gradient norm: {}".format(self.losses['gradient_norm'][-1]))
                print()
                # Plot images to track performance
                if self.plot_images: 
                    gen_imgs = self.genImages()
                    self.plotImages(gen_imgs)
                    
                        # noise = self.G.sample_latent(9).to(self.device)
                    
                    
                
                  
                   
        
            # Compute FID score, Generate images
            # if self.num_steps % 500 == 0:
            #     # if self.save_FID==True:
                    
            #     #     print('\nCalculate FID on {} generated Images'.format(self.FIDImages))
            #     #     tracked_FID = self.GenImgGetFID(self.FIDImages)
            #     #     self.tracked_info['FID'].append(tracked_FID)
            #         print()
            

        #Run FID score at the end of an Epoch    
        # if self.save_FID==True:
        #     # print('\nCompute EpochFID on {} generated Images'.format(self.FIDImages))
        #     # tracked_FID = self.GenImgGetFID(self.FIDImages)
        #     # self.tracked_info['EpochFID'].append(tracked_FID)
        #     print()
        
        self.EpochUpdateInfo() # Update informations
        self.DumpInfo() # save information about FID, Iterations...
        if self.saveModelsPerEpoch:
            self.save_model(self.checkpoint_folder) # save model generator and discriminator
        
        # if (self.epoch+1) % 2 == 0:
        #     if self.save_images: 
        #         gen_images = self.genImages()
        #         self.saveImages(gen_images,self.FIDPaths[0]) # save images per epoch to trace performance       
        

        if self.save_images: 
            gen_images = self.genImages()
            self.saveImages(gen_images,self.FIDPaths[0]) # save images per epoch to trace performance       


    def train(self, data_loader, epochs):
        ''' Train the network \n input: dataloader, epochs'''
        self.fixed_noise = self.G.sample_latent(9).to(self.device) # noise to generate images to save
        
        for epoch in range(epochs):
            print("\nEpoch {}".format(epoch))
            self.epoch = epoch
            start_time = time.time()
            self._train_epoch(data_loader)
            print('Epoch {} finished in {} minutes'.format(self.epoch, (time.time()-start_time)/60) )
        



    def sample_generator(self, num_samples):
        latent_samples = self.G.sample_latent(num_samples).to(self.device)
        generated_data = self.G(latent_samples)
        return generated_data

    
    
    
         
        
        
    def EpochUpdateInfo(self):
        '''Update LossD and LossG in tracked_info'''
        self.tracked_info['Epochs'] = self.epoch +1
        for k, v in self.losses.items():
            if k in self.tracked_info.keys():
                self.tracked_info[k] = v
                
        self.tracked_info['Iterations'] = self.num_steps
        
        # if self.save_FID==True:
        #     print('Calculate FID on {} Images'.format(self.FIDImages))
        #     tracked_FID = self.GenImgGetFID(self.FIDImages)
        #     self.tracked_info['FID'].append(tracked_FID)
        
        # if len(self.Fid_Values ) !=0:
        #     self.tracked_info['FID'].append( self.Fid_Values)
        # print(self.G.type)
            
        
    def DumpInfo(self):
        '''Dump tracked info'''
        if not os.path.isdir('./infos'):
            os.makedirs('./infos')
        info_path = './infos/'+ self.Generator_info + '_' + date + '.json'
        with open(info_path, 'w') as f:
          json.dump(self.tracked_info, f)
          
          
    def save_model(self, folder):
        '''Save D and G models \nfolder: where to save models'''
        
        # Save models
        gen_name = self.G.__class__.__name__
        disc_name = self.D.__class__.__name__
        
        gen_path = folder + gen_name
        disc_path = folder + disc_name
        
        torch.save(self.G.state_dict(), gen_path + '_epoch{}'.format(self.epoch) + '_BN-{}_SN-{}_SS-{}_ImgNorm-{}_{}'.format(
            self.G.batch_normed, hasattr(self.D, 'spectral_normed') and self.D.spectral_normed ==True, self.ssup, self.normalize, date) + '.pt')
        
        torch.save(self.D.state_dict(), disc_path + '_epoch{}'.format(self.epoch) + '_BN-{}_SN-{}_SS-{}_ImgNorm-{}_{}'.format(
            self.G.batch_normed, hasattr(self.D, 'spectral_normed') and self.D.spectral_normed ==True, self.ssup, self.normalize, date) + '.pt')
        




    def genImages(self):
        ''' Return a 3x3 grid of 9 images'''
        with torch.no_grad():
            self.G.eval()
            
            fake = self.G(self.fixed_noise)
            
            if fake.size(1) == 3:
                fake = fake
                # print(fake.shape)
            else:
                fake = fake[:,1:4,:,:]
            
            if self.normalize:
                imgs = vutils.make_grid(fake.detach().cpu().data, normalize=True, range=(-1,1), padding=2, nrow=3)
            else:
                imgs = vutils.make_grid(fake.detach().cpu().data, padding=2, nrow=3)
        self.G.train()
        
        return imgs
    
        
    def plotImages(self, imgs):
        '''Plot grid of images'''
        if not hasattr(self, 'fig'):
            self.fig= '_'
            plt.figure(figsize=(3,3))
        plt.ion()
        plt.axis("off")
        plt.title("Generated Images")
        plt.imshow(imgs.permute(1,2,0))
        plt.show()
        plt.pause(0.1)
            
        
    def saveImages(self, imgs, path):
        '''Save grid of images'''

        print('Images saved')
        plt.ioff()
        plt.axis("off")
        plt.title("Generated Images")
        imgs = imgs.permute(1,2,0)
        # images from [-1, 1] to [0, 1]
        # imgs = (imgs + 1) / 2
        img_path = path + '/Epoch{}'.format(self.epoch) + '.png'
        plt.imsave(img_path, imgs.numpy())
        # plt.close()
                
    
        
        
        
    # def saveImagePerEpoch(self, path):
    #     with torch.no_grad():
            
    #         self.G.eval()
    #         for n in range(self.ImagesPerEpoch):
    #             noise = self.G.sample_latent(1).to(self.device)
    #             fake = self.G(noise)
    #             if fake.size(1) == 3:
    #                 fake = fake[0].permute(1,2,0)
    #             else:
    #                 fake = fake[0,1:4,:,:].permute(1,2,0)
                    
    #             if self.normalize:
    #                 img = vutils.make_grid(fake.cpu().data, normalize=self.normalize, range=(-1,1))
    #             else:
    #                 img = vutils.make_grid(fake.cpu().data)
                
    #             img_path = path + '/Epoch{}'.format(self.epoch) +'_img' + str(n)   + '.png'
    #             plt.imsave(img_path, img.numpy())
                
    #     self.G.train()
        
                
    # Genrate images to calculate FID
    # def GenImgGetFID(self, sampled_images):
        
    #     generated_images_path, original_images_path = self.FIDPaths

    #     with  torch.no_grad():
    #         self.G.eval()
            
    #         for n in range(sampled_images):
    #             im = self.sample_generator(1)[0].detach().cpu()
                
    #             if im.size(0) ==4:
    #                 im = im[1:4,:,:]                
    #             im = np.transpose(im, (1,2,0))
    #             im = torch.squeeze(im)           
    #             img_path = generated_images_path + '/img' + str(n) + '.png'
      
    #             if torch.min(im) < 0 or torch.max(im) >1 :
    #               im = torch.clamp(im, min=0, max=1)
                  
    #             plt.imsave(img_path, im.numpy()  )
    #     self.G.train()
        
    #     if len(os.listdir(generated_images_path)) > 0:
    #         return self.FID(path=[original_images_path, generated_images_path])            