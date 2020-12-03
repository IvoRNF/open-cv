import numpy as np 
from ga import GeneticAlgorithm

logging = False 

def log(msg):
    global logging
    if not logging:
        return 
    print(msg) 

x = np.array([[0,0,255],[0,255,0],[0,0,255],[0,0,255]])
target = np.array([0,1,0,0])

log(x.shape)

weights = []
hidden_layer_sizes = [2,1]
output_layer_size = 2 

last_input_sz =  x.shape[1]
for neuron_count in [*hidden_layer_sizes,output_layer_size]: 
    weights_of_layer = np.random.uniform(low=-0.1,high=0.1,size=(last_input_sz,neuron_count))
    weights.append(weights_of_layer) 
    last_input_sz = neuron_count  

log('len weights')
log(len(weights))
log('input weights ')
log(weights[0])
log('first hidden layer weights ')
log(weights[1])
log('second hidden layer weights ')
log(weights[2])

log('mutiplield by the input weights')
sample = x[0]
y = np.dot(sample,weights[0])
log(y)
log(y.shape)

   
def multiply(msg,inpt,weights):
    log(msg)
    y = np.dot(inpt,weights)
    log(y)
    log(y.shape)
    return y

def softmax(inpt):
    return np.exp(inpt)/np.sum( np.exp(inpt) )    
def sigmoid(input_vl): 
    return 1.0/(1.0+np.exp(-1 * input_vl))    

def predict(inpt):
    y = multiply(msg='mutiplield by the input weights',inpt=sample,weights=weights[0]) 
    y = sigmoid(y)
    y = multiply(msg='mutiplield by the first hidden layer weights',inpt=y,weights=weights[1])
    y = sigmoid(y)
    y = multiply(msg='multiplield by then second hidden layer weights',inpt=y,weights=weights[2])
    return y 
def weights_flattened():
    global weights
    result = []
    for layer in weights: 
        for vl in layer.flat: 
           result.append(vl)
    return np.array(result)
def weights_unflattened(flattened):
    global weights
    result = [] 
    start_idx = 0
    for layer in weights: 
       wts_of_layer = flattened[start_idx: start_idx + layer.shape[0]*layer.shape[1] ] 
       start_idx += (len(wts_of_layer)) 
       wts_of_layer = np.array(wts_of_layer)
       wts_of_layer = np.reshape(wts_of_layer,newshape=layer.shape) 
       result.append(wts_of_layer) 
    return result    

print(weights)
print('...')
flattened = weights_flattened()

print('...')

print(flattened)

print('....')


print(weights_unflattened(flattened))