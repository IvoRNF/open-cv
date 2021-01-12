import torch 
from file_loader import FileLoader
import cv2 
import numpy as np
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from feature import pre_process
from torchvision import transforms
import torch.nn.functional as nnfunc
from torch import nn

class MyNNPyTorch(nn.Module):

    def __init__(self): 
        super().__init__()
        self.load_data()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=20,kernel_size=5,stride=1)
        self.conv2 = nn.Conv2d(in_channels=20,out_channels=50,kernel_size=5,stride=1)
        self.lin_in_feat = 50 * 29 * 13
        self.fc1 = nn.Linear(in_features=self.lin_in_feat,out_features=500)
        self.fc2 = nn.Linear(in_features=500,out_features=3)    

    def forward(self,x):
        x = nnfunc.relu(self.conv1(x))
        x = nnfunc.max_pool2d(x,2,2)
        x = nnfunc.relu(self.conv2(x))
        x = nnfunc.max_pool2d(x,2,2)
        x = x.view(-1,self.lin_in_feat)
        x = nnfunc.relu(self.fc1(x))
        x = self.fc2(x)
        return nnfunc.log_softmax(x,dim=1)
        
        
    def load_data(self):
        loader = FileLoader(dir_to_walk=r'C:\Users\Ivo Ribeiro\Documents\open-cv\datasets\captures')
        loader.load_files()
        self.class_names = loader.class_names
        x_data,y_data = self.convert_files_to_tensors(loader.files)
        x_val,y_val = self.convert_files_to_tensors(loader.files_test)
        self.train_ds = TensorDataset(x_data,y_data)
        self.val_ds = TensorDataset(x_val,y_val)
        batch_size = 6
        self.train_dl = DataLoader(self.train_ds,batch_size=batch_size)
        self.val_dl = DataLoader(self.val_ds,batch_size=batch_size)
          
    def convert_files_to_tensors(self,files):
        x_data = torch.tensor(())
        y_data = torch.tensor(())
        transformer = transforms.Compose([
            transforms.ToTensor()
        ])
        for row in files:
             dirs = row['imgs_per_class']
             for fname in dirs:
                 img = cv2.imread(fname,cv2.IMREAD_GRAYSCALE)
                 img = pre_process(img)
                 img = transformer(img)
                 x_data = torch.cat((x_data,img.unsqueeze(dim=1)),0)
                 y_data = torch.cat((y_data,torch.tensor([row['index']])),0)
        return (x_data,y_data)             
if __name__=='__main__':
    my_nn = MyNNPyTorch()
    for x,y in my_nn.train_dl:
        #inp = x[0][0].numpy()
        #print(inp.shape)
        out = my_nn(x)
        print(out)  
        break 
    print('ok')