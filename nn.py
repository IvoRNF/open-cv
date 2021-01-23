import numpy as np 
from ga import GeneticAlgorithm
import pickle 
import os


class MyNeuralNetwork:

    def __init__(self,inpt_layer_size,hidden_layer_sizes,output_layer_size,model_f_name,logging=True):
        self.model_f_name = model_f_name
        self.inpt_layer_size = inpt_layer_size 
        self.hidden_layer_sizes = hidden_layer_sizes 
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
        for neuron_count in [*self.hidden_layer_sizes,self.output_layer_size]: 
            weights_of_layer = np.random.uniform(low=-0.1,high=0.1,size=(last_input_sz,neuron_count))
            self.weights.append(weights_of_layer) 
            last_input_sz = neuron_count 

    def sigmoid(self,input_vl): 
        return 1.0/(1.0+np.exp(-1 * input_vl))  

    def sigmoid_derivative(self,s):
        return s * (1-s)    

    
        
    def forward(self,inpt):
        y = inpt
        for i in range(len(self.weights)):
            w = self.weights[i]
            b = self.biases[i]
            y = np.dot(y,w)  + b
            y = self.sigmoid(y)    
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
        result = [] 
        start_idx = 0
        for layer in self.weights: 
            wts_of_layer = flattened[start_idx: start_idx + layer.shape[0]*layer.shape[1] ] 
            start_idx += (len(wts_of_layer)) 
            wts_of_layer = np.array(wts_of_layer,dtype=np.float64)
            wts_of_layer = np.reshape(wts_of_layer,newshape=layer.shape) 
            result.append(wts_of_layer) 
        return result    
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
    def fitness_func(self,sol):   
        self.weights = self.weights_unflattened(sol)
        return self.eval_model()
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
    def save(self):
        with open(self.model_f_name,'wb') as f:
           mapa = {"weights":self.weights_flattened(),"bias":self.biases} 
           pickle.dump(mapa,f)
        print('saved model to file %s' % (self.model_f_name))        
    def train_ga(self,generations=90):
        flattened = self.weights_flattened()
        initial_solutions = np.random.uniform(low=-1.0,high=1.0,size=(20,len(flattened)))
        initial_solutions[0] = np.array(flattened,dtype=np.float64)
        ga = GeneticAlgorithm(
            solutions=initial_solutions, 
            num_parents_for_mating=4,
            generations=generations,
            fitness_func=self.fitness_func ,
            offspring_sz=4
        )
        ga.start()
        self.weights =  self.weights_unflattened(ga.solutions[0]) #best solution
        

if __name__ == '__main__':

    model_f_name = './datasets/my_nn77.pk'
    inpt_sz = 2
    hidden_szs = [2]
    outpt_sz = 2
    x_train = np.array([[0,255],[100,0],[0,255],[0,255],[255,0],[255,0],[255,0]])
    y_train = np.array([0,1,0,0,1,1,1])
    print('1 - train and evaluate\n2 - load and evaluate') 
    v = input()
    if v=='1':
        print('training')
        nn = MyNeuralNetwork(inpt_sz,hidden_szs,outpt_sz,model_f_name,True)
        nn.fit(x_train,y_train)
        nn.try_load()
        nn.train_ga(25)
        stats = nn.eval_model(ret_stats=True)
        print(stats)
        print('Save model ? 1=Y,2=N')
        v = input()
        if v=='1':
            nn.save()
    elif v=='2':    
        nn= MyNeuralNetwork(inpt_sz,hidden_szs,outpt_sz,model_f_name,True)
        nn.fit(x_train,y_train)
        nn.try_load()   
        print(nn.forward(np.array([0,255])))
        stats = nn.eval_model(ret_stats=True)
        print(stats)
     