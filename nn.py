import numpy as np 

inputs = np.array([4,5,7,10])

desired_outputs = np.array([8,10,14,20])

weights = np.array([0.05,1],dtype=np.float32) #bias and weight


def square_error(desired,predicted): 
    return  np.power(desired-predicted,2)

def relu(input_vl):
    if input_vl < 0: 
        return 0
    else: 
        return input_vl     
def sigmoid(input_vl):
    return 1/(1+np.power(np.e,-input_vl) )    

def update_weights(predicted , idx, learning_rate = 0.0001): 
    global desired_outputs
    global inputs
    global weights 
    der_magn = 0    #backpropagation
    der_magn = (inputs[idx] * (2 * (predicted - desired_outputs[idx])))
    if der_magn==0:
        return weights
    elif der_magn> 0:
      weights[1] = weights[1] - (learning_rate * np.abs(der_magn))  
    else:
      weights[1] = weights[1] + (learning_rate * np.abs(der_magn))  
    return weights


def training_loop(max_iterations=2000): 
    global inputs 
    global weights
    global desired_outputs
    abs_error = 1 
    idx = 0 
    i= 0 
    while (i < max_iterations) or (abs_error>=0.01)  : 
        z = weights[0] * 1 + weights[1] * inputs[idx]
        predicted = relu(z)
        error = square_error(desired_outputs[idx],predicted)
        abs_error = np.abs( error )
        weights = update_weights(predicted,idx)
        idx += 1 
        idx = idx % inputs.shape[0]
        i += 1  
    return error     

if __name__ == '__main__': 
    print('ok')
    error = training_loop()
    print('End training.')
    print('Weights ',weights)
    for input_vl in [20,30,40,50,2,5]: 
        z = weights[0] * 1 + weights[1] * input_vl
        predicted = relu(z)
        print('%.2f predicted as %.2f' % (input_vl,predicted))    