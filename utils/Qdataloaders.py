import os
import torch
import numpy as np
import pickle

from torchvision import datasets, transforms
import PIL
from PIL import Image
import matplotlib.pyplot as plt

# from multiprocessing import cpu_count



class Pad(object):
    def __init__(self):
        return
    
    def __call__(self, input):
        
        return(torch.nn.functional.pad(input, pad= (0,0,0,0,1,0), mode= 'constant', value= 0))
    

def preprocessing(quat_data, img_size, normalize):
        ''' quat_data: if data is quaternionic
            img_size: data reshape size (squared)
            normalize: if data has to be normalized in [0,1]
            
            '''
        R = transforms.Resize(img_size)
        C = transforms.CenterCrop(img_size)
        T = transforms.ToTensor()
        lista = []
        lista.extend([R, C, T])
        
        if quat_data:
            P = Pad()
            lista.append(P)
            if normalize:
                N = transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))
                lista.append(N)
        else:
            if normalize:
                N = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                lista.append(N)
        print('Preprocessing:\n',transforms.Compose(lista))
        return transforms.Compose(lista)




# def CelebA_dataloader(root, quat_data, img_size, normalize, batch_size, num_workers=cpu_count()):
#     """CelebA dataloader with resized and normalized images."""
#     name = 'CelebA'
#     dataset = datasets.ImageFolder(root=root,
#                                     transform=preprocessing(quat_data, img_size, normalize)
#                                     )
#     # Create the dataloader
#     train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
#                                               shuffle=True, num_workers=num_workers)


    
#     test_loader = '_'
    
#     return train_loader, test_loader, name






#-----------------------------------------------------------------------------------------------------------#

class ARRange(object):
    ''' Change values of the pixels of the images to match the output of the generator in [-1,1] '''
    def __init__(self, out_range, in_range=[0,255]):
        self.in_range = in_range
        self.out_range = out_range
        self.scale = (np.float32(self.out_range[1]) - np.float32(self.out_range[0])) / (np.float32(self.in_range[1]) - np.float32(self.in_range[0]))
        self.bias = (np.float32(self.out_range[0]) - np.float32(self.in_range[0]) * self.scale)
        
        # print(self.scale, self.bias)
    def __call__(self, input):

        return input * self.scale + self.bias


class To_Tensor_custom(object):
    ''' Change values of the pixels of the images to match the output of the generator in [-1,1] '''
    def __init__(self):

        pass
        
    def __call__(self, pic):
        # handle PIL Image
        # print(pic.mode)
        
        # if pic.mode == 'I':
        #     img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        # elif pic.mode == 'I;16':
        #     img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        # elif pic.mode == 'F':
        #     img = torch.from_numpy(np.array(pic, np.float32, copy=False))
        # elif pic.mode == '1':
        #     img = 255 * torch.from_numpy(np.array(pic, np.uint8, copy=False))
        # else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    
        img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
        # put it from HWC to CHW format
        img = img.permute((2, 0, 1)).contiguous()
        
        
        # print(input.shape)
        return img


# class Image_dataset():
def preprocessing2(quat_data, img_size, normalize):
        ''' quat_data: if data is quaternionic
            img_size: data reshape size (squared)
            normalize: if data has to be normalized
            
            '''
        R = transforms.Resize(img_size)
        C = transforms.CenterCrop(img_size)
        # T = transforms.ToTensor()
        # T = transforms.PILToTensor()
        T = To_Tensor_custom()
        lista = []
        lista.extend([ R, C, T])
        
        if quat_data:
            P = Pad()
            lista.append(P)
            
        if normalize:
            N = ARRange([-1,1])
            # N = transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))
        else:
            N = ARRange([0,1])
        lista.append(N)
       
        
        return transforms.Compose(lista)    
    
 

    
class CelebA_dataset(torch.utils.data.Dataset):

    def __init__(self, root_dir,  transform=None):
        self.root_dir = root_dir
        self.im_list = os.listdir(self.root_dir)
        self.transform = transform

        # print(self.im_list)
    
    def __len__(self):
        return len(self.im_list)
    
    def __getitem__(self, idx):
        im = Image.open(os.path.join(self.root_dir, self.im_list[idx]))
        # print(im)
        # im = np.array(im)*255
        # im = torch.from_numpy(np.array(im))
        # print(torch.max(im), torch.min(im))

        if self.transform:
            im = self.transform(im)

        return im, 'label'
    
    
 




def add_dim(img):
    print(img.size())
    return img.view(img.size(0), img.size(1), img.size(2), img.size(3), 1)


def preprocessing_HQ(quat_data, img_size, normalize, keep_4_channel=False):
        ''' quat_data: if data is quaternionic
            img_size: data reshape size (squared)
            normalize: if data has to be normalized
            
            '''
        lista = []
        T = To_Tensor_custom()
        lista.append(T)
        
        if quat_data:
            if keep_4_channel:
                lista.append(transforms.Lambda(add_dim))

            P = Pad()
            lista.append(P)
            
        if normalize:
            N = ARRange([-1,1])
            # N = transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))
        else:
            N = ARRange([0,1])
        lista.append(N)

            
       
        
        return transforms.Compose(lista)    
    
 

    
class CelebAHQ_dataset(torch.utils.data.Dataset):

    def __init__(self, root_dir,  transform=None):
        self.root_dir = root_dir
        self.im_list = os.listdir(self.root_dir)
        self.transform = transform

        # print(self.im_list)
    
    def __len__(self):
        return len(self.im_list)
    
    def __getitem__(self, idx):
        im = Image.open(os.path.join(self.root_dir, self.im_list[idx]))
        # print(im)
        # im = np.array(im)*255
        # im = torch.from_numpy(np.array(im))
        # print(torch.max(im), torch.min(im))

        if self.transform:
            im = self.transform(im)

        return im, 'label'
    
    
def CelebAHQ_dataloader(root, quat_data, img_size, normalize, batch_size, num_workers=10):
    """CelebA-HQ dataloader with resized and normalized images."""
    name = 'CelebA-HQ'
    print('Dataset:', name)

    dataset = CelebAHQ_dataset(root_dir=root,
                             transform=preprocessing_HQ(quat_data, img_size, normalize)
                             )
    print('Samples:', dataset.__len__())
    print()
    print('Preprocessing:\n', dataset.transform)
    # Create the dataloader
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                              shuffle=True, num_workers=num_workers)


    
    test_loader = '_'
    
    return train_loader, test_loader, name


def CelebA_dataloader2(root, quat_data, img_size, normalize, batch_size, num_workers=2):
    """CelebA dataloader with resized and normalized images."""
    name = 'CelebA'
    print('Dataset:', name)
    
    
    # print(root)
    dataset = CelebA_dataset(root_dir=root,
                             transform=preprocessing2(quat_data, img_size, normalize)
                             )
    print('Samples:', dataset.__len__())
    print()
    print('Preprocessing:\n', dataset.transform)
    # Create the dataloader
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                              shuffle=True, num_workers=num_workers)


    
    test_loader = '_'
    
    return train_loader, test_loader, name   


#----------------------------------------------------------------------------------------------------#

'''LSUN'''

def LSUN_dataloader(root, quat_data, img_size, normalize, batch_size, num_workers=2, train_class='bedroom_train'):
    """LSUN_Bedroom dataloader with resized and normalized images."""
    name = 'LSUN ' +' '.join(train_class.split('_')[:-1])
    print('Dataset:', name)
    
    dataset = datasets.LSUN(root= root, classes=['bedroom_train'],
            transform = preprocessing2(quat_data, img_size, normalize))

                             
    print('Samples:', dataset.__len__())
    print()
    print('Preprocessing:\n', dataset.transform)
    # Create the dataloader
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                              shuffle=True, num_workers=num_workers)


    
    test_loader = '_'
    
    return train_loader, test_loader, name   


#----------------------------------------------------------------------------------------------------#

''' CIFAR10 '''

# torchvision.datasets.CIFAR10(root: str, train: bool = True, transform: Union[Callable, NoneType] = None, target_transform: Union[Callable, NoneType] = None, download: bool = False) → None


def CIFAR10_dataloader(root, quat_data, img_size, normalize, batch_size, num_workers=2, eval=False):
    """CIFAR10 dataloader with resized and normalized images."""
    
    if not eval:
        name = 'CIFAR10'
        print('Dataset:', name)
        
        dataset = datasets.CIFAR10(root=root, download= True,
                                transform = preprocessing_HQ(quat_data, img_size, normalize)) 
        
        # dataset = datasets.LSUN(root= root, classes=['bedroom_train'],
        #         transform = preprocessing2(quat_data, img_size, normalize))

                                
        print('Samples:', dataset.__len__())
        print()
        print('Preprocessing:\n', dataset.transform)
        # Create the dataloader
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                shuffle=True, num_workers=num_workers)

        test_loader = '_'
        return train_loader, test_loader, name   

    test_root = "C:/Users/eleon/Documents/Dottorato/Code/QRotGAN/CIFAR/data/Test_FID_cifar"
    if eval:
        dataset_test = datasets.CIFAR10(root=test_root, download= True, train=False,
                                    transform = preprocessing_HQ(quat_data=False, img_size=32, normalize=False))
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False)

        # for i, (data, _) in enumerate(test_loader):
        #     # print(data.size())
        #     # print(data)
        #     # break
        #     plt.ioff()
        #     plt.axis("off")
        #     imgs = data[0].permute(1,2,0)
        #     img_path = test_root + str(i) + '.png'
        #     plt.imsave(img_path, imgs.numpy())



        # with open("C:/Users/eleon/Documents/Dottorato/Code/QRotGAN/CIFAR/data/Test_FID_cifar/cifar-10-batches-py/test_batch", 'rb') as fo:
        #     test_dict = pickle.load(fo, encoding='bytes')
        # for elem in test_dict:
        #     # print(elem.shape)
        #     print(elem)
        #     img = Image.fromarray(elem[1], 'RGB')
        #     img.save("C:/Users/eleon/Documents/Dottorato/Code/QRotGAN/CIFAR/data/Test_FID_cifar/test_cifar")


    


































# --------------------------------------- Dataloader for colab ------------------------------------------- #

import h5py
# from PIL import Image
import io

class dataset_h5(torch.utils.data.Dataset):
    def __init__(self, root,  transform=None):
        super(dataset_h5, self).__init__()
        
        self.transform = transform
        
        self.in_file = root
        # self.resize_dim = resize_dim
        self.len = len(list(h5py.File(self.in_file, 'r')['..']['data']['celebA_Train']['Train'].keys()))
 
    def open_hdf5(self):
        
        self.file = h5py.File(self.in_file, 'r')
        self.dataset = self.file['..']['data']['celebA_Train']['Train']
        self.keys = list(self.file['..']['data']['celebA_Train']['Train'].keys())
        self.len = len(self.keys)


    def __getitem__(self, index):
        if not hasattr(self, 'file'):
            self.open_hdf5()

        x = self.dataset[self.keys[index]]
        x =  Image.open(io.BytesIO(np.array(x)))
        
        # Preprocessing each image
        if self.transform is not None:
            x = self.transform(x)
            
        return x, '_'
 
    def __len__(self):
        return self.len 


def CelebA_colab_dataloader(root, quat_data, img_size, normalize, batch_size, num_workers=2):
    """CelebA dataloader for hdf5 dataset."""
    name = 'CelebA'
    # print(root)
    dataset = dataset_h5(root=root,
                             transform=preprocessing2(quat_data, img_size, normalize)
                             )
    # Create the dataloader
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                              shuffle=True, num_workers=num_workers)


    test_loader = '_'
    
    return train_loader, test_loader, name
    
# def get_QCelebA_dataloader(root, batch_size, img_size, num_workers=cpu_count()):
#     """CelebA dataloader with resized images."""
#     name = 'CelebA'
#     dataset = datasets.ImageFolder(root=root,
#                                     transform=transforms.Compose([
#                                     transforms.Resize(img_size),
#                                     transforms.CenterCrop(img_size),
#                                     transforms.ToTensor(),
#                                     Pad(),
#                                     # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#                                 ]))
#     # Create the dataloader
#     train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
#                                               shuffle=True, num_workers=num_workers)


    
#     test_loader = '_'
    
#     return train_loader, test_loader, name





# def get_CelebA_QGAN_dataloader(root, batch_size, img_size, num_workers=cpu_count()):
#     """CelebA dataloader for quaternion networks"""
#     name = 'Q_CelebA'
#     dataset = datasets.ImageFolder(root=root,
#                                     transform=transforms.Compose([
#                                     transforms.Resize(img_size),
#                                     transforms.CenterCrop(img_size),
#                                     transforms.ToTensor(),
#                                     # Pad(),
#                                     # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#                                 ]))
#     # Create the dataloader
#     train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
#                                               shuffle=True, num_workers=num_workers)


    
#     test_loader = '_'
    
#     return train_loader, test_loader, name




# def get_CelebA_Rot_dataloader(root, batch_size=128, img_size = 128, num_workers=cpu_count()):
#     """CelebA dataloader with (32, 32) sized images."""
#     name = 'CelebA'
#     dataset = datasets.ImageFolder(root=root,
#                                    transform=transforms.Compose([
#                                    transforms.Resize(146),
#                                    transforms.RandomCrop(128),
#                                    # Pad(),
#                                    transforms.ToTensor(),
#                                    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#                                ]))
#     # Create the dataloader
#     train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
#                                              shuffle=True, num_workers=num_workers)
    
#     # test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
#                                              # shuffle=True)


    
#     test_loader = '_'
    
#     return train_loader, test_loader, name






# class QCelebaDataset(torch.utils.data.Dataset):
    
#     def __init__(self, img_size, transform=None):
#         self.root_dir = './data/celeba/img_align_celeba/Train/Train_celeba'
        
#         self.im_list = os.listdir(self.root_dir)
#         # print(self.im_list)
#         self.resize_dim = (img_size,img_size)
#         if transform==None:
#             self.transform = transforms.Compose([
#                                         transforms.ToPILImage(float),
#                                         transforms.Resize(img_size),
#                                         transforms.CenterCrop(img_size),
                                        
#                                         transforms.ToTensor(),
#                                         ImagePad(),
#                                         # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#                                     ])
#         else:
#             self.transform = transform

#     def __len__(self):
#         return len(self.im_list)

#     def __getitem__(self, idx):
#         im = Image.open(os.path.join(self.root_dir, self.im_list[idx]))
#         im = np.array(im)
#         im = im / 255

#         if self.transform:
#             im = self.transform(im)

#         # Manipulation for quaternion net
#         # npad  = ((1, 0), (0, 0), (0, 0))
#         # im = np.pad(im, pad_width=npad, mode='constant', constant_values=0)
#         return im
    
    
    
# class CelebaDataset(torch.utils.data.Dataset):
    
#     def __init__(self, root_dir,img_size, transform=None):
#         self.root_dir = root_dir
#         self.im_list = os.listdir()
#         self.resize_dim = img_size
#         if transform==None:
#             self.transform = transforms.Compose([
#                                         transforms.Resize(img_size),
#                                         transforms.CenterCrop(img_size),
#                                         # ImagePad(),
#                                         transforms.ToTensor(),
#                                         # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#                                     ])
#         else:
#             self.transform = transform

#     def __len__(self):
#         return len(self.im_list)

#     def __getitem__(self, idx):
#         im = Image.open(os.path.join(self.root_dir, self.im_list[idx])).resize(self.resize_dim, resample=PIL.Image.NEAREST)
#         im = np.array(im)
#         im = im / 255

#         if self.transform:
#             im = self.transform(im)

#         # Manipulation for quaternion net
#         # npad  = ((1, 0), (0, 0), (0, 0))
#         # im = np.pad(im, pad_width=npad, mode='constant', constant_values=0)
#         return im
    
    
    
    
# def QCelebaLoader(batch_size=64, img_size = 64):
#     """CelebA dataloader with (32, 32) sized images."""
#     name = 'CelebA'
#     dataset = QCelebaDataset(img_size)
#     # Create the dataloader
#     train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
#                                              shuffle=True)
    
#     # test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
#                                              # shuffle=True)


    
#     test_loader = '_'
    
#     return train_loader, test_loader, name