import numpy as np 
import pickle 
import os
from feature import hot_encode_vect


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
        bias_count = 0 
        for layerConnWts in self.weights:  
                 bias_count += layerConnWts.shape[0]
        self.biases = np.random.uniform(low=-0.1,high=0.1,size=(bias_count))      

    def init_weights(self):
        w1 = np.random.uniform(low=-0.1,high=0.1,size=(self.hidden_layer_size,self.inpt_layer_size ))
        w2 = np.random.uniform(low=-0.1,high=0.1,size=(self.output_layer_size,self.hidden_layer_size ))
        self.weights =  [w1,w2]
          
    def sigmoid(self,input_vl): 
        return 1.0/(1.0+np.exp(-1 * input_vl))  

    def mseLoss(self,y,output):
        result = 0.5 * np.power(y - output,2)
        return np.sum(result)   

    
        
    def forward(self,inpt,doLog=False,doRetHiddenNeuronOutputs=False):
        y = np.array([])
        oldDoLog = self.logging
        self.logging = doLog
        j = 0
        hidden_neuron_outputs = np.array([])
        firstLayerIdx = 0
        lastLayerIdx = 1
        self.log('%s %s'%('1.start forward',y))
        for i in range(len(self.weights)):
            layerConnWts = self.weights[i]
            for wtsBatch in layerConnWts:
                w = wtsBatch
                b = self.biases[j]
                j+=1  
                if (i==firstLayerIdx):
                    nout = nn.neuron_output(inpt,wtsBatch,b)
                    hidden_neuron_outputs = np.append(hidden_neuron_outputs,nout)
                else:
                   y = np.append(y, nn.neuron_output(hidden_neuron_outputs,wtsBatch,b)  )              
        self.logging = oldDoLog
        if doRetHiddenNeuronOutputs:
            return y,hidden_neuron_outputs
        return y 
    def weights_flattened(self):
        result = []
        for layerConnWts in self.weights: 
            for wtsBatch in layerConnWts:
                for w in wtsBatch:
                    result.append(w)           
        return np.array(result,dtype=np.float64)
    def weights_unflattened(self,flattened):
        last = self.inpt_layer_size * self.hidden_layer_size
        w1 = np.array(flattened[0:last])
        w1 = w1.reshape(self.hidden_layer_size,self.inpt_layer_size)
        wtsCount = self.hidden_layer_size * self.output_layer_size
        w2 = np.array(flattened[last: last + wtsCount])
        w2 = w2.reshape(self.output_layer_size,self.hidden_layer_size)
        return [w1,w2]  
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
           #print(self.weights)
    def save(self):
        flattened = self.weights_flattened()
        #print(flattened)
        with open(self.model_f_name,'wb') as f:
           mapa = {"weights":flattened,"bias":self.biases} 
           pickle.dump(mapa,f)
        print('saved model to file %s' % (self.model_f_name))        
      
    def der_of_last_layer(self,y,output,der_of_weight):
        err_out_der =  output - y 
        out_outin_der = output * (1 - output) 
        return err_out_der * out_outin_der * der_of_weight 

    def backward(self,x,y,output,lr=0.001,hidden_neuron_outputs=None):
        #derivatives of sigmoid
        derivativesLastLayerByBatch = []
        derivatives = []
        for hout in hidden_neuron_outputs: 
            der = self.der_of_last_layer(y,output, hout * (1 - hout) )
            derivativesLastLayerByBatch.append(der)
      
        weightsBatchIdx=0
        lastWts  =  self.weights[1]
        
        lastWtsFlat = np.zeros(shape=(lastWts.shape[0] * lastWts.shape[1]),dtype=lastWts.dtype)
        start = weightsBatchIdx * lastWts.shape[1]
        lastWtsFlat[start : start + lastWts.shape[1]] =  lastWts[weightsBatchIdx][:]
            
        derivativesLastLayer = np.zeros( shape=lastWtsFlat.shape,dtype=lastWts.dtype )
        derivativesLastLayer[start : start + lastWts.shape[1] ] = derivativesLastLayerByBatch[:]
                
        for derLastLayer,w in zip(derivativesLastLayer,lastWtsFlat):
          for xN in x: 
              der = w * xN * derLastLayer
              derivatives.append(der)
        derivatives.extend(derivativesLastLayer)
        self.apply_derivatives(lr,derivatives)
        

    
    def train(self,epochs=2000,lr=0.5,sanityCheck = False):
        i = 0
        for i in range(epochs): 
            for x,y in zip(self.x_train,self.y_train):
                out,hidden_neuron_outputs = self.forward(x,doLog=False,doRetHiddenNeuronOutputs=True)
                err = self.mseLoss(y,out)
                self.log('err %.5f' % (err))
                self.log(f'out {out}')
                if self.output_layer_size==1:   
                   nn.backward(x,y,out,lr=lr,hidden_neuron_outputs = hidden_neuron_outputs)  
                else: 
                   nn.backward2(x,y,out,lr=lr,hidden_neuron_outputs = hidden_neuron_outputs) 
                   print(self.weights)   
                if sanityCheck:
                    break       
        print('trained epochs %d' % (i))      

    def backward2(self,x,y,output,lr=0.001,hidden_neuron_outputs=None):
        derivativesLastLayer = []
        deltasLastLayer = []
        for expectedN,outN in zip(y,output):
            delta = (-(expectedN - outN)) * (outN * (1 - outN) )
            for houtN in hidden_neuron_outputs:
               der = delta * houtN 
               derivativesLastLayer.append(der)
            deltasLastLayer.append(delta)
        lastWts = self.weights[1]      
        derivativesFirstLayer = []
        for wts,houtN in zip(lastWts.T,hidden_neuron_outputs): 
            soma = 0
            for w,delta in zip(wts,deltasLastLayer):
                soma += delta * w 
            for xN in x:
                der = soma * (houtN * (1-houtN)) * xN
                derivativesFirstLayer.append(der)    
        derivatives = []
        derivatives.extend(derivativesFirstLayer)
        derivatives.extend(derivativesLastLayer) 
        self.apply_derivatives(lr,derivatives)
    

    def apply_derivatives(self,lr,derivatives):
        newWeights = self.weights_flattened()
        for i in range(len(newWeights)):
            der = derivatives[i]
            newWeights[i] = newWeights[i] - (lr * der)
        self.weights = self.weights_unflattened(newWeights)        
    def neuron_output(self,x,wBacth,bias):
        result = 0
        for i in np.arange(x.shape[0]):
                result += x[i] * wBacth[i] + bias   
        return self.sigmoid(result)
         
if __name__ == '__main__':

    model_f_name = './datasets/my_nn000.pk'
    inpt_sz = 2
    hidden_sz = 2
    '''
    
    outpt_sz = 1
    x_train = np.array([[0.1,0.3]])
    y_train = np.array([1])
    
    '''
    outpt_sz = 2
    x_train = np.array([[0.05,0.1]])
    sz = 2 
    y_train = [[0.01,0.99]]
    
    print('1 - train and evaluate\n2 - load and evaluate') 
    nn = MyNeuralNetwork(inpt_sz,hidden_sz,outpt_sz,model_f_name,True)
    nn.fit(x_train,y_train)
    nn.try_load()
    v = input()    
    if v == '1':          
        print('training')
        nn.train(epochs=1000 * 10  ,lr=0.5,sanityCheck=False)            
        print('Save model ? 1=Y,2=N')
        inp = input()
        if inp=='1':
            nn.save()
    elif v == '2':       
        out = nn.forward(x_train[0],doLog=False)
        print(f'out {out}')
        err = nn.mseLoss(y_train[0],out)
        print('err %.5f' % (err))
        
       
        
    
     