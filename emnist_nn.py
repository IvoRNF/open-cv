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
import os

class EmnistNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.root_dir = './emnist'
        self.modelfname = './Emnist_model.pt'
        
        self.batch_size = 16
        self.test_sz = 0.15
        self.load_dataset()
        self.target_sz = 10
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=20,kernel_size=5,stride=1)
        self.conv2 = nn.Conv2d(in_channels=20,out_channels=50,kernel_size=5,stride=1)

        self.linear1_in_features = 50 * 4 * 4

        self.fc1 = nn.Linear(in_features=self.linear1_in_features,out_features=500)
        self.fc2 = nn.Linear(in_features=500,out_features=10 )

        self.loss_func = nn.MSELoss(reduction='sum') 
        self.opt = optim.SGD(self.parameters(),lr=1e-4)
        self.try_load() 
    def forward(self,x_):
        x = x_.float()
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
        print('3 - evaluate')  
        i = input()
        if i=='1':
           self.show_sample()
        elif i=='2':
           self.trainning_loop()
        elif i=='3':
           self.evaluate_acc()
    def evaluate_acc(self):
        true_positive_ct = 0
        samples_ct = len(self.test_ds)
        print('evaluating ...')
            
        for xb,yb in self.test_dl:
            y_ = self(xb)
            yb = self.prepare_yb(yb)
            for y_expected,y_predicted in zip(yb,y_):
                y_predicted_ = y_predicted.detach().numpy()
                idx = np.argmax(y_predicted_,axis=0)
                if y_expected[idx]==1:
                   true_positive_ct += 1
            
        print('test samples count %d ' % (samples_ct))
        print( 'model accuracy {:.2f}'.format(true_positive_ct / samples_ct) )
        print('true positives count %d ' % (true_positive_ct))
                
    def prepare_yb(self,yb):
        arr = yb.numpy()
        result = np.zeros(shape=(arr.shape[0],self.target_sz),dtype=np.float32)
        for i in np.arange(len(arr)):
            result[i][arr[i]] = 1
        result = torch.tensor(result)
        return result
    def load_dataset(self):
        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5), (0.5))])
        self.dataset = datasets.EMNIST(root=self.root_dir,split='digits',download=True,transform=transform)
        train_idxs, test_idxs = train_test_split(list(range(len(self.dataset))), test_size=self.test_sz,shuffle=True)
    
        self.test_ds = Subset(self.dataset,test_idxs)
        self.train_ds = Subset(self.dataset,train_idxs)
        self.train_dl = DataLoader(self.train_ds,batch_size=self.batch_size)
        self.test_dl = DataLoader(self.test_ds,batch_size=self.batch_size)
        print('total_sz = %d ,test_sz = %d , train_sz = %d' % (len(self.dataset),len(self.test_ds) ,len(self.train_ds)))
    def show_sample(self):
        imgs = self.dataset.data[0:10].numpy()        
        display(np.concatenate(imgs))
    def try_load(self):
        if os.path.exists(self.modelfname):
           self.load_state_dict(torch.load(self.modelfname)) 
           print('loaded model from file %s' % (self.modelfname))
    def trainning_loop(self,epochs=1,break_at_loss=5):
        has_breaked = False
        for epoch in np.arange(epochs):
            if has_breaked:
              break
            self.train()
            for xb,yb in self.train_dl:
                out = self(xb)
                yb = self.prepare_yb(yb)
                loss = self.loss_func(out,yb)
                loss.backward() #compute the gradients
                self.opt.step() #update the weights
                self.opt.zero_grad() #clear the gradients of batch
                loss_ = loss.item()
                if int(loss_ * 100) <= break_at_loss:
                   print('breaking training at loss = %.2f' % (loss_))
                   has_breaked = True 
                   break
                   
                print('training epoch %d,loss %.2f' % (epoch,loss_)) 
        print('saving the model...')  
        torch.save(self.state_dict(),self.modelfname)         
        print('saved.')
if __name__=='__main__':
    model = EmnistNN()
    model.start()
