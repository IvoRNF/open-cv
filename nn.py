import numpy as np 


class MyNeuralNetwork:

    def __init__(self,max_iterations=1000,learning_rate = 0.0001,activation_func='relu',hidden_layer_sizes=None, output_layer_size=0):
        self.max_iterations= max_iterations
        self.learning_rate = learning_rate
        self.activation_func = activation_func
        self.hidden_layer_sizes = hidden_layer_sizes
        self.output_layer_size = output_layer_size


    def isLinear(self): 
        return (self.hidden_layer_sizes is None)
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

           while(i < self.max_iterations): 
              input_ = self.inputs[idx]  
              for j in range( len(self.weights)-1 ): 
                  input_ = self.predict(input_ , j)
              input_ = input_ * self.weights[-1]
              
              idx += 1 
              idx = idx % self.inputs.shape[0]
              i += 1   

 
           return 
        while (i < self.max_iterations) or (abs_error>=0.01)  : 
            predicted = self.predict(self.inputs[idx])
            error = self.square_error(self.desired_outputs[idx][0],predicted[0])
            abs_error = np.abs( error )
            self.update_weights(predicted,idx)
            idx += 1 
            idx = idx % self.inputs.shape[0]
            i += 1   

    def predict(self,input_vl,weight_idx=None):         
        if weight_idx is None:
          arr = input_vl * self.weights
        else:
          arr =  input_vl * self.weights[i]
        func_name = self.activation_func
        func = getattr(self,func_name)
        return func(arr)
if __name__ == '__main__': 
    inputs_train = np.array([[3],[4],[7],[10]])
    outputs_train = np.array([[6],[8],[14],[20]])  
    nn = MyNeuralNetwork()#hidden_layer_sizes=np.array([150,60]),output_layer_size=4)
    nn.fit(x=inputs_train,y=outputs_train)
    print('Weights ',nn.weights)
    inputs_test = np.array([[20],[30],[40],[50],[2],[5]])
    predicted_vls = nn.predict(inputs_test)
    i = 0 
    for predicted in predicted_vls:     
      print('%.2f predicted as %.2f' % (inputs_test[i],np.round(predicted)))
      i += 1 

    