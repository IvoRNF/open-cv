import numpy as np 


class MyNeuralNetwork:

    def __init__(self,max_iterations=1000,learning_rate = 0.0001,activation_func='relu'):
        self.max_iterations= max_iterations
        self.learning_rate = learning_rate
        self.activation_func = activation_func

    def fit(self,x:np.ndarray,y:np.ndarray):
        self.inputs = x 
        self.desired_outputs = y
        self.weights = np.random.uniform(low=-0.1,high=0.1,size=(2)) 
        self.weights[0]=0.05#bias  
        self.train() 

    def square_error(self,desired,predicted): 
         return  np.power(desired-predicted,2)

    def relu(self,input_vl):
        if input_vl < 0: 
            return 0
        else: 
            return input_vl     
    def sigmoid(self,input_vl):
        return 1/(1+np.power(np.e,-input_vl) )    

    def update_weights(self,predicted , idx): 
        der_magn = 0    #backpropagation
        der_magn = (self.inputs[idx] * (2 * (predicted - self.desired_outputs[idx])))
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
            error = self.square_error(self.desired_outputs[idx],predicted)
            abs_error = np.abs( error )
            self.update_weights(predicted,idx)
            idx += 1 
            idx = idx % self.inputs.shape[0]
            i += 1     
    def predict(self,input_vl):         
        z = self.weights[0] * 1 + self.weights[1] * input_vl
        func_name = self.activation_func
        func = getattr(self,func_name)
        return func(z)
if __name__ == '__main__': 
    inputs = np.array([3,4,7,10])
    outputs = np.array([9,12,21,30])  
    nn = MyNeuralNetwork()
    nn.fit(x=inputs,y=outputs)
    print('Weights ',nn.weights)
    for input_vl in [20,30,40,50,2,5]:
        predicted = nn.predict(input_vl)
        print('%.2f predicted as %.2f' % (input_vl,np.round(predicted)))