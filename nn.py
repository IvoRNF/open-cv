import numpy as np 


class MyNeuralNetwork:

    def __init__(self,max_iterations=1000,learning_rate = 0.0001,activation_func='softmax',hidden_layer_sizes=None, output_layer_size=0):
        self.max_iterations= max_iterations
        self.learning_rate = learning_rate
        self.activation_func = activation_func
        self.hidden_layer_sizes = hidden_layer_sizes
        self.output_layer_size = output_layer_size


    def isLinear(self): 
        return (self.hidden_layer_sizes is None)
    
    def softmax(self,inpt):
       return np.exp(inpt)/np.sum( np.exp(inpt) )

    def fit(self,x:np.ndarray,y:np.ndarray):
        self.inputs = x 
        self.desired_outputs = y
        if self.isLinear(): 
           self.weights = np.random.uniform(low=-0.1,high=0.1,size=( self.inputs.shape[1] )) 
        else: 
          last_input_sz =  self.inputs.shape[1]
          self.weights = []
          for neuron_count in [*self.hidden_layer_sizes,self.output_layer_size]: 
              weights_of_layer = np.random.uniform(low=-0.1,high=0.1,size=(last_input_sz,neuron_count))
              self.weights.append(weights_of_layer) 
              last_input_sz = neuron_count          
        self.train() 

    def square_error(self,desired,predicted): 
         return  np.power(desired-predicted,2)

    def relu(self,input_vl):
        result = input_vl
        result[result < 0] = 0 
        return result 
    def sigmoid(self,input_vl): #nao testado
        return 1.0/(1+np.power(np.e,-1 * input_vl) )    

    def update_weights(self,predicted , idx): 
        der_magn = 0    #backpropagation
        der_magn = (self.inputs[idx][0] * (2 * (predicted[0] - self.desired_outputs[idx][0])))
        if der_magn==0:
            return 
        elif der_magn> 0:
            self.weights = self.weights - (self.learning_rate * np.abs(der_magn))  
        else:
            self.weights = self.weights + (self.learning_rate * np.abs(der_magn))  
    def train(self): 
        abs_error = 1 
        idx = 0 
        i= 0 
        if not self.isLinear(): 
           '''
           while(i < self.max_iterations): 
              input_ = self.inputs[idx]  
              for j in range( len(self.weights)-1 ): 
                  input_ = self.predict(input_ , j)
              input_ = input_ * self.weights[-1]
              
              idx += 1 
              idx = idx % self.inputs.shape[0]
              i += 1   
            '''
 
           return 
        while (i < self.max_iterations) or (abs_error>=0.01)  : 
            predicted = self.predict(self.inputs[idx])
            error = self.square_error(self.desired_outputs[idx][0],predicted[0])
            abs_error = np.abs( error )
            self.update_weights(predicted,idx)
            idx += 1 
            idx = idx % self.inputs.shape[0]
            i += 1   

    def is_odd_or_zero(self,n):
        if n==0:
           return True  
        return (n % 2)!=0
    def predict(self,input_vl,weight_idx=None):  
               
        func_name = self.activation_func
        func = getattr(self,func_name)
        if self.isLinear():
            if weight_idx is None:
               arr = input_vl * self.weights
            else:
               arr =  input_vl * self.weights[weight_idx]
            return func(arr)
        out_in = input_vl
        for i in range(len(self.weights)):
            wts = self.weights[i]
            out_in = np.dot(out_in,wts)
            print('multiply')
            if self.is_odd_or_zero(i): 
               out_in = func(out_in)
               print('activate')
        return out_in       
if __name__ == '__main__': 
    inputs_train = np.array([[0,0,255],[0,255,0],[0,0,255],[0,0,255]])
    outputs_train = np.array([0,1,0,0])  
    nn = MyNeuralNetwork(hidden_layer_sizes=np.array([3,3]),output_layer_size=2)
    nn.fit(x=inputs_train,y=outputs_train)
    print('Weights ',nn.weights)
    inputs_test = np.array([[0,0,255]])
    predicted_vls = nn.predict(inputs_test)
    print('predicted')
    print(predicted_vls) 

    