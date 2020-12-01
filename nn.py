import numpy as np 


class MyNeuralNetwork:

    def __init__(self,max_iterations=1000,learning_rate = 0.0001,activation_func='softmax',n_hidden_layers=1, output_layer_size=0):
        self.max_iterations= max_iterations
        self.learning_rate = learning_rate
        self.activation_func = activation_func
        self.n_hidden_layers = n_hidden_layers
        self.output_layer_size = output_layer_size
        self.logging = False
 
    def softmax(self,inpt):
       return np.exp(inpt)/np.sum( np.exp(inpt) )

    def fit(self,x:np.ndarray,y:np.ndarray):
        self.inputs = x 
        self.desired_outputs = y
        neuron_count =  self.inputs.shape[1]
        self.weights = []
        for _ in range(self.n_hidden_layers): 
            weights_of_layer = np.random.uniform(low=-0.1,high=0.1,size=(neuron_count  + 1))
            self.weights.append([weights_of_layer]) 
        output_layer_wts = []    
        for _ in range(self.output_layer_size): 
            weights_of_layer = np.random.uniform(low=-0.1,high=0.1,size=(self.n_hidden_layers  + 1))
            output_layer_wts.append(weights_of_layer)
        self.weights.append(output_layer_wts)   
        self.neurons_metadata = []
        for layer in self.weights: 
            arr = []
            for _ in layer:
                arr.append({})
            self.neurons_metadata.append(arr) 

    def loss_func(self,desired,predicted): 
         return  0.5 * np.power(desired-predicted,2)
    def relu(self,input_vl):
        result = input_vl
        if result < 0:
            result = 0 
        return result 
    def sigmoid(self,input_vl): 
        return 1.0/(1.0+np.exp(-1 * input_vl)) 


    def backward_propagate_error(self,expected):
        for i in reversed(range(len(self.weights))):
            layer = self.weights[i]
            errors = []
            if i != len(self.weights)-1:
                for j in range(len(layer)):
                    err = 0.0
                    for nextLayer in self.weights[i + 1]:
                        err += (nextLayer[j] * self.neurons_metadata[i+1][j]['delta'])
                    errors.append(err)
            else:
                for j in range(len(layer)):
                    neuron = layer[j]
                    loss = self.loss_func(expected[j] , neuron[i][j]['output'])
                    errors.append( loss )
            for j in range(len(layer)):
                neuron = layer[j]
                self.neurons_metadata[i][j]['delta'] = 
                      errors[j] * (neuron[i][j]['output'] * (1 - neuron[i][j]['output']))  #transfer derivative     
    def update_weights(self,predicted , idx): 
        '''
        der_magn = 0    #backpropagation
        der_magn = (self.inputs[idx][0] * (2 * (predicted[0] - self.desired_outputs[idx][0])))
        if der_magn==0:
            return 
        elif der_magn> 0:
            self.weights = self.weights - (self.learning_rate * np.abs(der_magn))  
        else:
            self.weights = self.weights + (self.learning_rate * np.abs(der_magn))  
        '''    
    def log(self,msg):
        if self.logging:
           print(msg)
    
    
    def train(self): 
        idx = 0 
        i= 0 
        self.log('training')
        while (i < self.max_iterations): 
              predicted = self.predict(self.inputs[idx])
              predicted = np.max(predicted)
              desired = np.max(self.desired_outputs[idx])
              error = self.loss_func(desired,predicted)
              self.log('sample')
              self.log(self.inputs[idx])
              self.log('predicted')
              self.log(predicted)
              self.log('error')
              self.log(error)
              self.update_weights(predicted,idx)
              idx += 1 
              idx = idx % self.inputs.shape[0]
              i += 1   
    def predict(self,input_vl):  
        func_name = self.activation_func
        func = getattr(self,func_name)
        out_in = input_vl
        for i in range(len(self.weights)): 
            layer_w = self.weights[i]
            new_inputs = []
            for j in range(len(layer_w)):
                neuron = layer_w[j] 
                activ = neuron[-1] # bias 
                for k in range(neuron.shape[0]-1):
                   activ += neuron[k] * out_in[k]
                activ = func(activ)   
                new_inputs.append(activ)
                self.neurons_metadata[i][j]['output'] = activ 
            out_in = new_inputs
        return out_in    
if __name__ == '__main__': 
    inputs_train = np.array([[0,255],[255,0],[0,255],[0,255]])
    outputs_train = np.array([[1,0],[0,1],[1,0],[1,0]])  
    nn = MyNeuralNetwork(n_hidden_layers=1,output_layer_size=2,max_iterations=2000,activation_func='sigmoid')
    nn.fit(x=inputs_train,y=outputs_train)
    #nn.train()
    inputs_test = np.array([255,0])
    predicted_vls = nn.predict(inputs_test)
    print(nn.weights)
    print(predicted_vls) 
    nn.backward_propagate_error([0,1])
    print(nn.neurons_metadata) 
    



    