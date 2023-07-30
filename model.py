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
    def __init__(self, imgChannels=1, featureDim=32*20*20, zDim=256, zDim2= 2):
        super(AE, self).__init__()

        # Initializing the 2 convolutional layers and 2 full-connected layers for the encoder
        self.encConv1 = nn.Conv2d(imgChannels, 16, 5)
        self.encConv2 = nn.Conv2d(16, 32, 5)
        self.encFC1 = nn.Linear(featureDim, zDim)
        self.encFC12 = nn.Linear(zDim, zDim2)

        # Initializing the fully-connected layer and 2 convolutional layers for decoder
        self.decFC1 = nn.Linear(zDim2, zDim)
        self.decFC2 = nn.Linear(zDim, featureDim)
        self.decConv1 = nn.ConvTranspose2d(32, 16, 5)
        self.decConv2 = nn.ConvTranspose2d(16, imgChannels, 5)

    def encoder(self, x):

        # Input is fed into 2 convolutional layers sequentially
        # The output feature map are fed into 2 fully-connected layers to predict mean (mu) and variance (logVar)
        # Mu and logVar are used for generating middle representation z and KL divergence loss
        x = f.relu(self.encConv1(x))
        x = f.relu(self.encConv2(x))
        x = x.view(-1, 32*20*20)
        z = self.encFC12(f.relu(self.encFC1(x)))
        #logVar = self.encFC22(F.relu(self.encFC2(x)))
        return z

    def reparameterize(self, mu, logVar):

        #Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar/2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decoder(self, z):

        # z is fed back into a fully-connected layers and then into two transpose convolutional layers
        # The generated output is the same size of the original input
        x = f.relu(self.decFC2(f.relu(self.decFC1(z))))
        x = x.view(-1, 32, 20, 20)
        x = f.relu(self.decConv1(x))
        x = torch.sigmoid(self.decConv2(x))
        return x

    def forward(self, x1):
       encode_1 = self.encoder(x1)
       #encode_2 = self.encoder(x2)
       out_1 = self.decoder(encode_1)
       #out_2 = self.decoder(encode_2)
       return out_1