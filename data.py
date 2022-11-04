from torch.utils.data import Dataset
from PIL import Image
import torch
import os
import numpy as np
import torchvision
import torchvision.transforms as transforms
import myutils

class ImageDataset(Dataset):
    def __init__(self, root, size,crop_size,augmentation=0):

        self.root = root
        self.augmentation = augmentation
        self.crop_size = crop_size
        self.size = size
        self.gt = sorted(os.listdir(root + "/gt"))
        self.imgs = sorted(os.listdir(root + "/input"))

    def __getitem__(self, index):   # get images to dataloader

        label = Image.open(self.root + self.gt[index % len(self.imgs)])
        if self.crop_size != None:
            label = myutils.crop_center(label, self.crop_size, self.crop_size)  # crop image if the images are not the same size
        label = label.resize(size = self.size)  # change size

        img = Image.open(self.img[index % len(self.imgs)])
        if self.crop_size != None:
            img = myutils.crop_center(label, self.crop_size, self.crop_size)
        img = img.resize(size = self.size)
        
        transform = transforms.Compose([
            transforms.ToTensor(),  #### use this !!!
        ])

        img = transform(img)
        label = transform(label)
        
        imgs = {"img": img, "label": label}   # create imgs dictionary

        return imgs

    def __len__(self):  # if error num_sampler should be positive -> because Dataset not yet Downloaded
        return len(self.imgs)