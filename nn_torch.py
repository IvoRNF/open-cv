import torch 
from file_loader import FileLoader
import cv2 
import numpy as np
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from feature import pre_process
from torchvision import transforms
import torch.nn.functional as nnfunc
from torch import nn,optim
import os

class MyNNPyTorch(nn.Module):

    def __init__(self,training=False): 
        super().__init__()
        self.training = training
        self.train_dl = None
        self.val_dl = None
        self.normalize = False
        self.batch_size = 6
        self.modelfname = r'C:\Users\Ivo Ribeiro\Documents\open-cv\anns\torch_model.pt'
        self.imgs_foldername = r'C:\Users\Ivo Ribeiro\Documents\open-cv\datasets\captures'   
        self.class_names = self.load_class_names(self.imgs_foldername)
        if training:
           self.load_data()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=20,kernel_size=5,stride=1)
        self.conv2 = nn.Conv2d(in_channels=20,out_channels=50,kernel_size=5,stride=1)
        self.lin_in_feat = 50 * 29 * 13
        self.fc1 = nn.Linear(in_features=self.lin_in_feat,out_features=500)
        self.fc2 = nn.Linear(in_features=500,out_features=len(self.class_names))    
        self.loss_func = nn.NLLLoss(reduction='sum') 
        self.opt = optim.Adam(self.parameters(),lr=1e-4)
        self.try_load_weights()
    def forward(self,x):
        x = nnfunc.relu(self.conv1(x))
        x = nnfunc.max_pool2d(x,2,2)
        x = nnfunc.relu(self.conv2(x))
        x = nnfunc.max_pool2d(x,2,2)
        x = x.view(-1,self.lin_in_feat)
        x = nnfunc.relu(self.fc1(x))
        x = self.fc2(x)
        return nnfunc.log_softmax(x,dim=1)
        
    def get_loss(self,predicted,expected):
        return self.loss_func(predicted,expected)

    def load_class_names(self,folder_base):
        fnames = os.listdir(folder_base)
        dir_names = [fname for fname in fnames if os.path.isdir(os.path.join(folder_base,fname))]
        return dir_names   

    def try_load_weights(self):
        if os.path.exists(self.modelfname):
           self.load_state_dict(torch.load(self.modelfname)) 
           print('loaded model from file %s' % (self.modelfname))

    def load_data(self):
        if self.val_dl is not None: 
            return (self.train_dl,self.val_dl)
        
        print('loading files...')
        loader = FileLoader(dir_to_walk=self.imgs_foldername)
        loader.load_files()
        self.train_dl = None
        if self.training:
           x_data,y_data = self.convert_files_to_tensors(loader.files)
           self.train_ds = TensorDataset(x_data,y_data)         
           self.train_dl = DataLoader(self.train_ds,batch_size=self.batch_size)
        x_val,y_val = self.convert_files_to_tensors(loader.files_test)
        self.val_ds = TensorDataset(x_val,y_val)
        self.val_dl = DataLoader(self.val_ds,batch_size=self.batch_size)
        print('files loaded.')
        return (self.train_dl,self.val_dl)
    def train_epochs(self,epochs=3):
        for epoch in np.arange(epochs):
            self.train()
            for xb,yb in self.train_dl:
                out = self(xb)
                loss = self.get_loss(out,yb)
                loss.backward() #compute the gradients
                self.opt.step() #update the weights
                self.opt.zero_grad() #clear the gradients of batch
                print('training epoch %d,loss %.2f' % (epoch,loss.item())) 
        print('saving the model...')  
        torch.save(self.state_dict(),self.modelfname)         
        print('saved.')  
    
    
    def get_mean_std(self,imgs):
        means = np.zeros(shape=(imgs.shape[0]))  
        stds =  np.zeros(shape=(imgs.shape[0]))
        for i in np.arange(imgs.shape[0]):
           img = imgs[i]
           m = np.mean(img,axis=(0,1))
           std = np.std(img,axis=(0,1))
           means[i] = m    
           stds[i] = std
        my_mean = np.mean(means)
        my_std = np.std(stds)
        return (my_mean,my_std)
        
    def convert_files_to_tensors(self,files):
        x_data = torch.tensor(())
        y_data = torch.tensor((),dtype=torch.int64)
        lbls = []
        imgs = []
        for row in files:
             dirs = row['imgs_per_class']
             for fname in dirs:
                 img = cv2.imread(fname,cv2.IMREAD_GRAYSCALE)
                 img = pre_process(img)
                 imgs.append(img)
                 lbls.append(row['index'])  
        imgs = np.array(imgs)   
        lbls = np.array(lbls) 
        nseed = 6 
        np.random.seed(nseed)
        np.random.shuffle(imgs)
        np.random.seed(nseed)
        np.random.shuffle(lbls)
        my_transforms = [transforms.ToTensor()]
        if self.normalize: 
           my_mean,my_std  = self.get_mean_std(imgs)
           my_transforms.append(transforms.Normalize([my_mean],[my_std]))
        transformer = transforms.Compose(my_transforms)         
        for img,y in zip(imgs,lbls):          
            x = transformer(img)
            x_data = torch.cat((x_data,x.unsqueeze(dim=1)),0)            
            y_data = torch.cat((y_data,torch.tensor([y],dtype=torch.int64)),0)
        return (x_data,y_data) 
    def evaluate_model(self):
        self.eval()
        _,val_dl = self.load_data()
        acc = 0
        sz = 0
        for xb,yb in val_dl:
            with torch.no_grad():
                sz += xb.shape[0]
                predicted = self(xb)
                predicted = predicted.numpy()
                out = np.argmax(predicted,axis=1)
                ybnp = yb.numpy()
                for j in np.arange(ybnp.shape[0]):
                    if ybnp[j]==out[j]:
                        print('%d validating %s %s' % (out[j],self.class_names[out[j]],predicted[j]))
                        acc += 1                       
        print('evaluation results , val(%.2f) acc %.2f' % (sz,(acc/sz)))    
                 
def evaluate_model():
    my_nn = MyNNPyTorch(training=False)
    my_nn.evaluate_model()
if __name__=='__main__':

    print('1 - train and evaluate\n2 - load and evaluate')
    v = input()
    if v=='1':
        my_nn = MyNNPyTorch(training=True)
        print('start train')
        my_nn.train_epochs()
        my_nn.evaluate_model()
        print('ok')
    elif v=='2':   
        evaluate_model()