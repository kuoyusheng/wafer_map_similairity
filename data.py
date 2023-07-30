from typing import Callable, Optional
from google.cloud import storage
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def get_mnist_data(path:str):
    train_data = datasets.MNIST(root=path, train = True, download=True)


# def get_training_dataloader(
#             training_data:datasets,
#             batch_size:int = 128, 
#             shuffle:bool = True)->DataLoader:
#       return DataLoader(dataset=training_data,batch_size=batch_size, shuffle=shuffle)
#NUM_WORKERS = os.cpu_count()
def get_training_dataloader(
        train_data_path:str ='~/cloudML/myMLtest/data',
        batch_size:int=128, 
        shuffle:bool=True)->DataLoader:
        
        train_data = datasets.MNIST(root=train_data_path, 
                                download=False,
                                train=True, 
                                transform=transforms.ToTensor()
                            )
        return DataLoader(dataset=train_data, 
                      batch_size=batch_size,
                      shuffle=shuffle)

#get_training_dataloader()
if __name__ == "__main__":
      get_mnist_data('data')
