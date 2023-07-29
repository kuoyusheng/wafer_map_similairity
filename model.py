import torch
import torch.nn as nn
import torch.nn.functional as f
#from torchvision import datasets, transforms 
import torch.optim as optim
from torch.autograd import Variable
#from torch.utils import data
#import matplotlib.pyplot as plt
#import random 
import numpy as np

"""
A simple classical VAE model 
"""

class AE(nn.Module):
    def __init__(self, imgChannel = 1, featureDim= 32*20*20, zDim = 256, zDim2 = 2):
        super(AE, self).__init__()
        
        # Enlisting all required layer components
        self.encConv1=nn.Conv2d(imgChannel, 16, 5)
        self.encConv2=nn.Conv2d(16,32,5)
        self.encFC1=nn.Linear(featureDim)
        self.encFC2=nn.Linear(zDim, zDim2)

        # Decoder part
        self.decFC1=nn.Linear(zDim2, zDim)
        self.decFC2=nn.Linear(zDim, featureDim)
        self.deConv1=nn.ConvTranspose2d(32,16,5)
        self.deConv2=nn.ConvTranspose2d(16,imgChannel,5)

    def encoder(self, x):
        x = f.relu(self.encConv1(x))
        x = f.relu(self.encConv2(x))
        x = x.view(-1, 32*20*20)
        z = self.encFC12(f.relu(self.encFC1(x)))
        return z
    
    def decoder(self, z):
        x=f.relu(self.decFC1(x))
        x=f.relu(self.decFC2(x))
        x=x.view(-1,32,20,20)
        x=f.relu(self.deConv1(x))
        x = f.relu(self.deConv2(x))
        x= torch.sigmoid(x)
        return x
    
