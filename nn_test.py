import numpy as np 
from ga import GeneticAlgorithm
import pickle 
import os 
logging = True 
weights = []
biases = []
hidden_layer_sizes = [2]
output_layer_size = 2 
model_f_name = './datasets/my_nn77.pk'
inpt_layer_size = 3

def log(msg=None):
    global logging
    if not logging:
        return 
    if msg is None:
      print()
    else:       
      print(msg) 
def init_bias():
    global biases
    biases = np.random.uniform(low=-0.1,high=0.1,size=(len(weights)))      

def init_weights():
    global hidden_layer_sizes
    global output_layer_size
    global weights
    global inpt_layer_size
    last_input_sz =  inpt_layer_size
    for neuron_count in [*hidden_layer_sizes,output_layer_size]: 
        weights_of_layer = np.random.uniform(low=-0.1,high=0.1,size=(last_input_sz,neuron_count))
        weights.append(weights_of_layer) 
        last_input_sz = neuron_count  
def softmax(inpt):
    return np.exp(inpt)/np.sum( np.exp(inpt) )    
def sigmoid(input_vl): 
    return 1.0/(1.0+np.exp(-1 * input_vl))  

def predict(inpt):
    global weights
    global biases 
    y = inpt
    for i in range(len(weights)):
        w = weights[i]
        b = biases[i]
        y = np.dot(y,w)  + b
        y = sigmoid(y)
    return y 
def weights_flattened():
    global weights
    result = []
    for layer in weights: 
        for vl in layer.flat: 
           result.append(vl)
    return np.array(result,dtype=np.float64)
def weights_unflattened(flattened):
    global weights
    if len(weights)==0:
        init_weights()
    result = [] 
    start_idx = 0
    for layer in weights: 
       wts_of_layer = flattened[start_idx: start_idx + layer.shape[0]*layer.shape[1] ] 
       start_idx += (len(wts_of_layer)) 
       wts_of_layer = np.array(wts_of_layer,dtype=np.float64)
       wts_of_layer = np.reshape(wts_of_layer,newshape=layer.shape) 
       result.append(wts_of_layer) 
    return result    

def eval_model(ret_stats=False):
    x_test = [[0,0,255],[0,100,0],[0,0,255],[0,0,255],[0,255,0],[0,255,0],[0,255,0]]
    y_test = [0,1,0,0,1,1,1]
    acc = 0
    c_correct = []
    probs = []
    for x,y in zip(x_test,y_test): 
        arr = predict(x)
        label = np.argmax(arr)
        if(label == y):    
           c_correct.append(1)
           probs.append(arr)
           prob = arr[label]
           if prob > 0:        
             acc += prob   
    acc = (acc / len(y_test))
    if ret_stats:
        return {'acc_float': acc, 'acc_int':len(c_correct)/len(y_test),'probs':[ (y_test[x],probs[x]) for x in np.arange(len(y_test)) ]}         
    return acc
def fitness_func(sol):
    global weights
    weights = weights_unflattened(sol)
    return eval_model()
if __name__ == '__main__':
    if not os.path.exists(model_f_name):
        log('training')
        logging = False
        init_weights()
        init_bias()
        flattened = weights_flattened()
        initial_solutions = np.random.uniform(low=-2.0,high=4.0,size=(10,len(flattened)))
        initial_solutions[0] = np.array(flattened,dtype=np.float64)
        ga = GeneticAlgorithm(
            solutions=initial_solutions, 
            num_parents_for_mating=2,
            generations=100,
            fitness_func=fitness_func 
        )
        ga.start()
        logging = True
        with open(model_f_name,'wb') as f:
           mapa = {"weights":ga.solutions[0],"bias":biases} 
           pickle.dump(mapa,f)
        log('saved model to file %s' % (model_f_name))
    else:    
        wts = None
        with open(model_f_name,'rb') as f:
           mapa = pickle.load(f)
           wts = mapa['weights']
           biases = mapa['bias']
        log('loaded model from file')   
        weights = weights_unflattened(wts)   
        print(predict([0,0,255]))
        stats = eval_model(ret_stats=True)
        log(stats)