import torch 
from file_loader import FileLoader
import cv2 
import numpy as np
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from feature import pre_process
from torchvision import transforms

class MyNNPyTorch():

    def __init__(self): 
        self.load_data()
    def load_data(self):
        loader = FileLoader(dir_to_walk=r'C:\Users\Ivo Ribeiro\Documents\open-cv\datasets\captures')
        loader.load_files()
        self.class_names = loader.class_names
        x_data,y_data = self.convert_files_to_tensors(loader.files)
        x_val,y_val = self.convert_files_to_tensors(loader.files_test)
        #print(x_data.shape)
        #print(x_val.shape)
        #print(y_data.shape)
        #print(y_val.shape)
        self.train_ds = TensorDataset(x_data,y_data)
        self.val_ds = TensorDataset(x_val,y_val)
        batch_size = 6
        self.train_dl = DataLoader(self.train_ds,batch_size=batch_size)
        self.val_dl = DataLoader(self.val_ds,batch_size=batch_size)
          
        for xb,yb in self.train_dl: 
            print(type(xb[0]))
            print(xb.shape)
            print(type(xb[0].numpy()))
            break
         
        #print(len(self.train_dl))
        #print(len(self.val_dl))
         
        
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
    MyNNPyTorch()
