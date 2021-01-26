import numpy as np 
import pickle 
import os


class MyNeuralNetwork:

    def __init__(self,inpt_layer_size,hidden_layer_size,output_layer_size,model_f_name,logging=True):
        self.model_f_name = model_f_name
        self.inpt_layer_size = inpt_layer_size 
        self.hidden_layer_size = hidden_layer_size
        self.output_layer_size = output_layer_size
        self.logging = logging
        self.weights = []
        self.biases = []
 
    def log(self,msg=None):
        if not self.logging:
            return 
        if msg is None:
           print()
        else:       
           print(msg) 

    def init_bias(self):
        self.biases = np.random.uniform(low=-0.1,high=0.1,size=(len(self.weights)))      

    def init_weights(self):
        last_input_sz =  self.inpt_layer_size
        for neuron_count in [self.hidden_layer_size,self.output_layer_size]: 
            weights_of_layer = np.random.uniform(low=-0.1,high=0.1,size=(last_input_sz,neuron_count))
            self.weights.append(weights_of_layer) 
            last_input_sz = neuron_count 

    def sigmoid(self,input_vl): 
        return 1.0/(1.0+np.exp(-1 * input_vl))  

    def mseLoss(self,y,output):
        return np.power(y - output,2)    
        
    def forward(self,inpt,doLog=False):
        y = inpt
        oldDoLog = self.logging
        self.logging = doLog
        for i in range(len(self.weights)):
            w = self.weights[i]
            b = self.biases[i]
            self.log('%s %s'%('1.start forward',y))
            y = np.dot(y,w)  + b
            self.log('%s %s'%('2.dot',y))
            y = self.sigmoid(y)    
            self.log('%s %s'%('3.sigmoid',y))
        self.logging = oldDoLog
        return y 
    def weights_flattened(self):
        result = []
        for layer in self.weights: 
            for vl in layer.flat: 
              result.append(vl)
        return np.array(result,dtype=np.float64)
    def weights_unflattened(self,flattened):
        if len(self.weights)==0:
            self.init_weights()
        '''
        result = []
        start_idx = 0
        for layer in self.weights: 
            wts_of_layer = flattened[start_idx: start_idx + layer.shape[0]*layer.shape[1] ] 
            start_idx += (len(wts_of_layer)) 
            wts_of_layer = np.array(wts_of_layer,dtype=np.float64)
            wts_of_layer = np.reshape(wts_of_layer,newshape=layer.shape) 
            result.append(wts_of_layer) 
        return result
        '''
        w1 = np.zeros(shape=self.inpt_layer_size)
        w2 = np.zeros(shape=self.hidden_layer_size)
        w3 = np.array([]) 

        w1[:] = flattened[0:self.inpt_layer_size]
        last = self.inpt_layer_size+self.hidden_layer_size
        w2[:] = flattened[self.inpt_layer_size:last]      
        for i in np.arange(last,len(flattened)):
            w3 = np.append(w3,flattened[i])
        return [w1,w2,w3]    
    def fit(self,x,y):
       self.x_train = x 
       self.y_train = y    

    def eval_model(self,ret_stats=False):
        acc = 0
        c_correct = []
        probs = []
        for x,y in zip(self.x_train,self.y_train): 
            arr = self.forward(x)
            label = np.argmax(arr)
            probs.append(arr)
            if(label == y):    
                c_correct.append(1)
                prob = arr[label]
                if prob > 0: 
                    acc += prob   
        acc = (acc / len(self.y_train))
        if ret_stats:
            return {'acc_float': acc, 'acc_int':len(c_correct)/len(self.y_train),'probs':[ (self.y_train[x],probs[x]) for x in np.arange(len(self.y_train)) ]}         
        return acc
    def try_load(self):
        modelf_exist = os.path.exists(self.model_f_name)
        if not modelf_exist:
           self.init_weights()
           self.init_bias()      
           return  
        with open(self.model_f_name,'rb') as f:
           mapa = pickle.load(f)
           self.weights = self.weights_unflattened(mapa['weights'])
           self.biases = mapa['bias']
           print('loaded model from file')  
           print(self.weights)
    def save(self):
        flattened = self.weights_flattened()
        print(flattened)
        with open(self.model_f_name,'wb') as f:
           mapa = {"weights":flattened,"bias":self.biases} 
           pickle.dump(mapa,f)
        print('saved model to file %s' % (self.model_f_name))        
    
    def der_of_last_layer(self,y,output,der_of_weight):
        err_out_der =  output - y 
        out_outin_der = output * (1 - output) 
        return err_out_der * out_outin_der * der_of_weight
  
    def der_of_middle_layer(self,x,y,output,w):
        err_out_der =  output - y 
        out_outin_der = output * (1 - output) 
        flattened = self.weights_flattened()
        w6 = flattened[6-1]
        #w3 = self.weights[1][0]
        #w4 = self.weights[1][1]
        outin_h2out_der = w6
        h2out_h2in_der =  self.neuron_output(x,1)
        h2out_h2in_der = (h2out_h2in_der * (1 - h2out_h2in_der)) 
        h2in_w = w 
        #print('err_out_der %.4f out_outin_der %.4f outin_h2out_der %.4f h2out_h2in_der  %.4f h2in_w4 %.4f' % (err_out_der ,out_outin_der,outin_h2out_der,h2out_h2in_der,h2in_w4))
        return err_out_der * out_outin_der * outin_h2out_der * h2out_h2in_der  * h2in_w
    
    def der_of_first_weights(self,x,y,output,h1in_w_der):
        err_out_der =  output - y 
        out_outin_der = output * (1 - output)
        flattened = self.weights_flattened() 
        w5 = flattened[5-1]
        outin_h1out = w5 
        h1out_h1in_der =   self.neuron_output(x,0) 
        h1out_h1in_der = (h1out_h1in_der * (1 - h1out_h1in_der))   
        #print('err_out_der %.4f out_outin_der %.4f outin_h1out %.4f h1out_h1in_der  %.4f h1in_w1_der %.4f' 
        #% (err_out_der ,out_outin_der , outin_h1out , h1out_h1in_der , h1in_w_der))
        return err_out_der * out_outin_der * outin_h1out * h1out_h1in_der * h1in_w_der
        
    def neuron_output(self,x,wi):
        wts = self.weights[wi]
        result = 0
        for i in np.arange(x.shape[0]):
            result += x[i] * wts[i]
        result = result + self.biases[wi]
        return self.sigmoid(result)
    def backward(self,x,y,output,lr=0.001):
        #derivatives of sigmoid
        h1out = self.neuron_output(x,0)
        outin_w5_der = h1out 
        err_w5_der = self.der_of_last_layer(y,output,outin_w5_der)
        h2out = self.neuron_output(x,1)
        outin_w6_der = h2out 
        err_w6_der = self.der_of_last_layer(y,output,outin_w6_der)
        flattened = self.weights_flattened()
        w4 = flattened[4-1]
        #w3 = self.weights[1][0]
        x0 = x[0]
        x1 = x[1]
        err_w4_der = self.der_of_middle_layer(x,y,output,w4) #w4 ?
        
        err_w3_der = self.der_of_middle_layer(x,y,output,x0)   
        err_w1_der = self.der_of_first_weights(x,y,output,x0)
        err_w2_der = self.der_of_first_weights(x,y,output,x1)
        
        ders = [err_w1_der,err_w2_der,err_w3_der,err_w4_der,err_w5_der,err_w6_der]
        k = 0 
        for i in np.arange( len(self.weights) ):
            for j in np.arange( self.weights[i].shape[0] ):
                der = ders[k]
                self.weights[i][j] = self.weights[i][j] - (lr * der)
                k += 1
        #print('err_w6_der %.5f outin_w5_der %.5f err_w4_der %.5f err_w3_der %.5f err_w1_der %.5f err_w2_der %.5f' 
        #   % (err_w6_der,err_w5_der,err_w4_der,err_w3_der,err_w1_der,err_w2_der))
    def train(self,epochs=2000,lr=0.8):
        i = 0
        for i in range(epochs):  
            out = self.forward(self.x_train[0],doLog=False)
            print('out %.5f' % (out))
            err = self.mseLoss(self.y_train[0],out)
            print('err %.5f' % (err))
            if out == self.y_train[0]:
                break 
            nn.backward(self.x_train[0],self.y_train[0],out,lr=lr)  
        print('trained epochs %d' % (i))       
if __name__ == '__main__':

    model_f_name = './datasets/my_nn79.pk'
    inpt_sz = 2
    hidden_sz = 2
    outpt_sz = 1
    x_train = np.array([[0.1,0.3]])
    y_train = np.array([0.03])
    print('1 - train and evaluate\n2 - load and evaluate') 
    nn = MyNeuralNetwork(inpt_sz,hidden_sz,outpt_sz,model_f_name,True)
    nn.fit(x_train,y_train)
    nn.try_load()
    v = input()    
    if v == '1':    
        nn.weights = [np.array([0.5,0.1]),
                        np.array([0.62,0.2]),
                        np.array([-0.2,0.3]),   ]
        nn.biases =  np.array([0.4,-0.1,1.83])   
    
        print('training')
        nn.train(lr=0.8,epochs=2000)            
        print('Save model ? 1=Y,2=N')
        inp = input()
        if inp=='1':
            nn.save()
    elif v == '2':       
        out = nn.forward(x_train[0],doLog=True)
        print('out %.5f' % (out))     
        err = nn.mseLoss(y_train[0],out)
        print('err %.5f' % (err))
       
        
    
     