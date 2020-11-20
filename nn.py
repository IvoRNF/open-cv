import numpy as np 


class MyNeuralNetwork:

    def __init__(self,max_iterations=1000,learning_rate = 0.0001,activation_func='relu',hidden_layer_sizes=None, output_layer_size=0):
        self.max_iterations= max_iterations
        self.learning_rate = learning_rate
        self.activation_func = activation_func
        self.hidden_layer_sizes = hidden_layer_sizes
        self.output_layer_size = output_layer_size

    def fit(self,x:np.ndarray,y:np.ndarray):
        self.inputs = x 
        self.desired_outputs = y
        if self.hidden_layer_sizes is None: #no hidden layer 
           self.weights = np.random.uniform(low=-0.1,high=0.1,size=( self.inputs.shape[1]+1 )) 
           self.weights[0]=0.05#bias  
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
    def sigmoid(self,input_vl):
        return 1.0/(1+np.power(np.e,-1 * input_vl) )    

    def update_weights(self,predicted , idx): 
        der_magn = 0    #backpropagation
        der_magn = (self.inputs[idx][0] * (2 * (predicted - self.desired_outputs[idx][0])))
        if der_magn==0:
            return 
        elif der_magn> 0:
            self.weights[1] = self.weights[1] - (self.learning_rate * np.abs(der_magn))  
        else:
            self.weights[1] = self.weights[1] + (self.learning_rate * np.abs(der_magn))  
    def train(self): 
        abs_error = 1 
        idx = 0 
        i= 0 
        while (i < self.max_iterations) or (abs_error>=0.01)  : 
            predicted = self.predict(self.inputs[idx])
            error = self.square_error(self.desired_outputs[idx][0],predicted)
            abs_error = np.abs( error )
            self.update_weights(predicted,idx)
            idx += 1 
            idx = idx % self.inputs.shape[0]
            i += 1     
    def predict(self,input_vl):         
        z = self.weights[0] * 1 + self.weights[1] * input_vl[0]
        func_name = self.activation_func
        func = getattr(self,func_name)
        return func(np.array([z]))
if __name__ == '__main__': 
    inputs = np.array([[3],[4],[7],[10]])
    outputs = np.array([[9],[12],[21],[30]])  
    nn = MyNeuralNetwork()#hidden_layer_sizes=np.array([150,60]),output_layer_size=4)
    nn.fit(x=inputs,y=outputs)
    print('Weights ',nn.weights)
    for input_vl in [[20],[30],[40],[50],[2],[5]]:
        predicted = nn.predict(input_vl)
        print('%.2f predicted as %.2f' % (input_vl[0],np.round(predicted)))
    