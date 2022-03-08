import numpy as np


''' solving diferential equations MIT 1806 course '''

A = np.array([
               [0,1],
               [1,0]
             ])


eigen_values,eigen_vectors = np.linalg.eig(A)

eigen_vectors = np.array([[1,1],[1,-1]])
print('eigen values')
print(eigen_values)
print('eigen vectors')
print(eigen_vectors)

u0 = np.array([4,2])
C = np.array([3,1])
def solution(t = 0):
    result = np.zeros(shape=C.shape)
    for i in np.arange(len(C)):
       result += (C[i] * (np.e ** (eigen_values[i] *t))) * eigen_vectors[i]            
    return result
           

print('solution =')
sol = solution(0)

print(sol)
#[4,2] 
