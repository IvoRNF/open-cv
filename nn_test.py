import numpy as np 
from ga import GeneticAlgorithm
import pickle 
import os 
logging = False 

def log(msg):
    global logging
    if not logging:
        return 
    print(msg) 

x = np.array([[0,0,255],[0,255,0],[0,0,255],[0,0,255]],dtype=np.float64)
target = np.array([0,1,0,0],dtype=np.float64)

weights = []
hidden_layer_sizes = [2]
output_layer_size = 2 
model_f_name = './datasets/my_nn77.pk'

last_input_sz =  x.shape[1]
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
    y = inpt
    for i in range(len(weights)):
        w = weights[i]
        y = np.dot(y,w)
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
    result = [] 
    start_idx = 0
    for layer in weights: 
       wts_of_layer = flattened[start_idx: start_idx + layer.shape[0]*layer.shape[1] ] 
       start_idx += (len(wts_of_layer)) 
       wts_of_layer = np.array(wts_of_layer,dtype=np.float64)
       wts_of_layer = np.reshape(wts_of_layer,newshape=layer.shape) 
       result.append(wts_of_layer) 
    return result    

def eval_model():
    x_test = [[0,0,255],[0,100,0],[0,0,255],[0,0,255],[0,255,0],[0,255,0],[0,255,0]]
    y_test = [0,1,0,0,1,1,1]
    acc = 0
    for x,y in zip(x_test,y_test): 
        arr = predict(x)
        label = np.argmax(arr)
        if(label == y): 
           prob = arr[label]
           if prob > 0:        
             acc += prob   
    return (acc / len(y_test))
def fitness_func(sol):
    global weights
    weights = weights_unflattened(sol)
    return eval_model()
if __name__ == '__main__':
    if not os.path.exists(model_f_name):
        print('training')
        flattened = weights_flattened()
        initial_solutions = np.random.uniform(low=-4.0,high=4.0,size=(8,len(flattened)))
        initial_solutions[0] = np.array(flattened,dtype=np.float64)
        ga = GeneticAlgorithm(
            solutions=initial_solutions, 
            num_parents_for_mating=2,
            generations=500,
            fitness_func=fitness_func 
        )
        ga.start()
        wts = ga.solutions[0] #best solution 
        with open(model_f_name,'wb') as f:
           pickle.dump(wts,f)
        print('saved model to file %s' % (model_f_name))
    else:    
        wts = None
        with open(model_f_name,'rb') as f:
           wts = pickle.load(f)
        print('loaded model from file')   
        weights = weights_unflattened(wts)   
        acc = eval_model()
        print('acc %.2f' % (acc))