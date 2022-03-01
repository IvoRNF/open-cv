import torch
from torch import nn,optim
import torch.nn.functional as nnfunc
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from util import display
import numpy as np
from sklearn.model_selection import train_test_split

class EmnistNN(nn.Module):

    def __init__(self):
        self.root_dir = './emnist'
        self.batch_size = 16
        self.test_sz = 0.15
        self.load_dataset()
    def forward(self,x):
        pass
    def start(self):
        print('1 - show sample')
        print('2 - train')

        i = input()
        if i=='1':
           self.show_sample()
        elif i=='2':
           pass
    def load_dataset(self):
        self.dataset = datasets.EMNIST(root=self.root_dir,split='byclass',download=True)
        train_idxs, test_idxs = train_test_split(list(range(len(self.dataset))), test_size=self.test_sz,shuffle=True)
    
        self.test_ds = Subset(self.dataset,test_idxs)
        self.train_ds = Subset(self.dataset,train_idxs)
        self.train_dl = DataLoader(self.train_ds,batch_size=self.batch_size)
        self.test_dl = DataLoader(self.test_ds,batch_size=self.batch_size)
        print('total_sz = %d ,test_sz = %d , train_sz = %d' % (len(self.dataset),len(self.test_ds) ,len(self.train_ds)))
        
    def show_sample(self):
        imgs = self.dataset.data[0:10].numpy()        
        display(np.concatenate(imgs))
if __name__=='__main__':
    model = EmnistNN()
    model.start()
