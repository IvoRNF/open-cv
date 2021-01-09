import torch 
from file_loader import FileLoader
import cv2 
import numpy as np

class MyNNPyTorch(FileLoader):

    def __init__(self):
        super().__init__(dir_to_walk=r'C:\Users\Ivo Ribeiro\Documents\open-cv\datasets\captures')
        
        self.load_data()
    def load_data(self):
        self.load_files()
        x_data,y_data = self.convert_files_to_tensors(self.files)
        x_val,y_val = self.convert_files_to_tensors(self.files_test)
        print(x_data.shape)
        print(y_data.shape)  
        print(x_val.shape)
        print(y_val.shape)  
        
    def convert_files_to_tensors(self,files):
        x_data = []
        y_data = []
        for row in files:
             dirs = row['imgs_per_class']
             for fname in dirs:
                 img = cv2.imread(fname,cv2.IMREAD_GRAYSCALE)
                 x_data.append([img]) 
                 y_data.append(row['index'])
        x_data = np.array(x_data) # to numpy 
        y_data = np.array(y_data)
        
        x_data = torch.tensor(x_data) # to tensor 
        y_data = torch.tensor(y_data)
        return (x_data,y_data)             
if __name__=='__main__':
    MyNNPyTorch()
