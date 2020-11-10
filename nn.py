import numpy as np 

inputs = np.array([4,5,7,10])

desired_outputs = np.array([8,10,14,20])

max_output = np.max(desired_outputs)

weights = np.array([0.05,.1]) #bias and weight


def prediction_error(desired,expected): 
    return np.abs(np.mean( desired-expected  ))

def activation_function(input_vl):
    global max_output
    if input_vl > max_output: 
        return max_output 
    else: 
      return input_vl      
def update_weights(predicted , idx, learning_rate = 0.00001): 
    global desired_outputs
    global inputs
    global weights 
    weights = weights + learning_rate * (desired_outputs[idx] - predicted)* inputs[idx]
    return weights


def training_loop(max_iterations=9000): 
    global inputs 
    global weights
    global desired_outputs
    error = 1 
    idx = 0 
    i= 0 
    while((i < max_iterations) or (error>=0.01)): 
        z = weights[0] * 1 + weights[1] * inputs[idx]
        predicted = activation_function(z)
        error = prediction_error(desired_outputs[idx],predicted)
        weights = update_weights(predicted,idx)
        idx += 1 
        idx = idx % inputs.shape[0]
        i += 1  
    return error     

if __name__ == '__main__': 
    error = training_loop()
    print('End training.')
    print('Weights ',weights)
    #new_inputs = np.array([25,60,65,50])
    #new_desired_outputs = np.array([50,120,130,100])
    #max_output = np.max(new_desired_outputs)
    #new_inputs = [4,5,7,10]
    for input_vl in inputs: 
        z = weights[0] * 1 + weights[1] * input_vl
        predicted = activation_function(z)
        print('%.2f predicted as %.2f' % (input_vl,predicted))    