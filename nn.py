import numpy as np 


class MyNeuralNetwork:

    def __init__(self,max_iterations=1000,learning_rate = 0.0001,activation_func='softmax',n_hidden_layers=1, output_layer_size=0):
        self.max_iterations= max_iterations
        self.learning_rate = learning_rate
        self.activation_func = activation_func
        self.n_hidden_layers = n_hidden_layers
        self.output_layer_size = output_layer_size
        self.logging = True
 
    def softmax(self,inpt):
       return np.exp(inpt)/np.sum( np.exp(inpt) )

    def fit(self,x:np.ndarray,y:np.ndarray):
        self.inputs = x 
        self.desired_outputs = y
        neuron_count =  self.inputs.shape[1]
        self.weights = []
        for _ in range(self.n_hidden_layers): 
            weights_of_layer = np.random.uniform(low=0,high=0.1,size=(neuron_count  + 1))
            self.weights.append([weights_of_layer]) 
        output_layer_wts = []    
        for _ in range(self.output_layer_size): 
            weights_of_layer = np.random.uniform(low=0,high=0.1,size=(self.n_hidden_layers  + 1))
            output_layer_wts.append(weights_of_layer)
        self.weights.append(output_layer_wts)   
        self.neurons_metadata = []
        for layer in self.weights: 
            arr = []
            for _ in layer:
                arr.append({})
            self.neurons_metadata.append(arr)
        self.train()     

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
                    for k in range( len(self.weights[i + 1]) ):
                        neur = self.weights[i + 1][k]
                        err += (neur[j] * self.neurons_metadata[i+1][k]['delta'])
                    errors.append(err)
            else:
                for j in range(len(layer)):   
                    output = self.neurons_metadata[i][j]['output']
                    loss = expected[j] - output
                    errors.append( loss )
            for j in range(len(layer)):
                output = self.neurons_metadata[i][j]['output']
                #transfer derivative
                self.neurons_metadata[i][j]['delta'] = errors[j] * output * (1 - output)       

    def update_weights(self,inpt):
        for i in range(len(self.weights)):
            layer = self.weights[i]
            inputs = inpt#[:-1] # remove o bias ?
            if i != 0:
                previous_layer = self.weights[i - 1]
                inputs = [ self.neurons_metadata[i][j]['output'] for j in range(len(previous_layer))]
            for j in range(len(layer)):
                delta = self.neurons_metadata[i][j]['delta']
                for k in range(len(inputs)):
                    self.weights[i][j][k] += self.learning_rate * delta * inputs[k]
                self.weights[i][j][-1] += self.learning_rate * delta  
        
    def log(self,msg):
        if self.logging:
           print(msg)
    def train(self): 
        i= 0 
        self.log('training')
        while (i < self.max_iterations): 
              for inpt,desired_outpt in zip(self.inputs,self.desired_outputs):
                  self.predict(inpt)
                  self.backward_propagate_error(desired_outpt)
                  self.update_weights(inpt)
              i += 1   
        self.log('trained')      
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
    inputs_train = np.array([[0,255],[255,0],[255,0],[255,0],[0,255],[0,255]])
    outputs_train = np.array([[1,0],[0,1],[0,1],[0,1],[1,0],[1,0]])  
    nn = MyNeuralNetwork(n_hidden_layers=1,output_layer_size=2,max_iterations=30000,activation_func='sigmoid',learning_rate=0.7)
    nn.fit(x=inputs_train,y=outputs_train)


    for inpt,outpt in zip(inputs_train,outputs_train): 
        predicted_vls = nn.predict(inpt)
        predicted_num = np.argmax(predicted_vls)
        expected_num = np.argmax(outpt)
        if predicted_num != expected_num:
          print('error')
        print('%d predicted as %d ' % (expected_num,predicted_num))
        print(predicted_vls)
    



    