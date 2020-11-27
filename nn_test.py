import numpy as np 

logging = False 

def log(msg):
    global logging
    if not logging:
        return 
    print(msg) 

x = np.array([[0,0,255],[0,255,0],[0,0,255],[0,0,255]])

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

def predict(inpt):
    y = multiply(msg='mutiplield by the input weights',inpt=sample,weights=weights[0]) 
    y = multiply(msg='mutiplield by the first hidden layer weights',inpt=y,weights=weights[1])
    y = multiply(msg='multiplield by then second hidden layer weights',inpt=y,weights=weights[2])
    return y 
sample = x[0]
y = predict(sample)
print(y)    


