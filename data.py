from typing import Callable, Optional
from google.cloud import storage
import pandas as pd
import torch
from torchvision import datasets, transforms, vision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader



#NUM_WORKERS = os.cpu_count()
def get_training_dataloader(
        train_data_path:str,
        batch_size:int=128, 
        shuffle:bool=True)->DataLoader:
        
        train_data = datasets.MNIST(root=train_data_path, 
                                train=True, 
                                transform=transforms.ToTensor()
                            )
        return DataLoader(dataset=train_data, 
                      batch_size=batch_size,
                      shuffle=shuffle)
