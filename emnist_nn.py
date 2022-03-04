import torch
from torch import nn,optim
import torch.nn.functional as nnfunc
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from util import display
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision import transforms

class EmnistNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.root_dir = './emnist'
        self.batch_size = 16
        self.test_sz = 0.15
        self.load_dataset()
        self.target_sz = 10
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=20,kernel_size=5,stride=1)
        self.conv2 = nn.Conv2d(in_channels=20,out_channels=50,kernel_size=5,stride=1)

        self.linear1_in_features = 50 * 4 * 4

        self.fc1 = nn.Linear(in_features=self.linear1_in_features,out_features=500)
        self.fc2 = nn.Linear(in_features=500,out_features=self.target_sz)

        self.loss_func = nn.MSELoss(reduction='sum') 
        self.opt = optim.SGD(self.parameters(),lr=1e-4)
        
    def forward(self,x_):

        x = torch.unsqueeze(x_,dim=1)
        x = x.float()
        x = self.conv1(x)
        x = nnfunc.relu(x)
        x = nnfunc.max_pool2d(x,2,2)
        x = self.conv2(x)
        x = nnfunc.relu(x)
        x = nnfunc.max_pool2d(x,2,2)
        x = x.view(-1,self.linear1_in_features)
        x = nnfunc.relu(self.fc1(x))
        x = self.fc2(x)
        x = nnfunc.softmax(x,dim=1)
        return x
    def start(self):
        print('1 - show sample')
        print('2 - train')

        i = input()
        if i=='1':
           self.show_sample()
        elif i=='2':
           imgs = self.dataset.data[0:1]
           print(self(imgs).detach().numpy().astype(np.float32))
    def load_dataset(self):
        self.dataset = datasets.EMNIST(root=self.root_dir,split='byclass',download=True,transform=transforms.Compose(transforms.ToTensor()))
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
