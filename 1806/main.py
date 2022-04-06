import numpy as np
import cv2

'''  MIT 1806 course '''

''' solving diferential equations 
  example 
  u0 = np.array([4,2])

  sol = diff_solution(A=np.array([
               [0,1],
               [1,0]
             ]),u0=u0,t=0,scal_eig_v=np.sqrt(2))

  print(sol)

  [4,2] expected at t=0 

'''
def diff_solution(A ,u0, t = 0,scal_eig_v=0):

    
    eigen_values,eigen_vectors = np.linalg.eig(A)

    if scal_eig_v != 0:
       eigen_vectors = eigen_vectors * scal_eig_v
     
    C = np.linalg.solve(eigen_vectors.T,u0)
    result = np.zeros(shape=C.shape)
    for i in np.arange(len(C)):
       result += (C[i] * (np.e ** (eigen_values[i] *t))) * eigen_vectors[i]            
    return result

'''
  returns  array of projections matrices 
  if matrix is symetric , the sum of these projections equals input A
'''

def projections_matrices(A,eigen_values,eigen_vectors):
    
    shape = (len(eigen_values),*eigen_vectors.shape)
    result = np.zeros(shape=shape,dtype=np.float32)
    for i in np.arange(len(eigen_values)):
        vl = eigen_values[i]
        vect = np.array([[*eigen_vectors[i]]])
        len_ = np.sqrt(np.dot(vect,vect.T)[0][0])
        vect = vect / len_
        matrix = ((vect * vl) * vect.T) 
        result[i] = matrix
    return result    
        
    

def degrees_to_radians(degrees):
    return degrees * np.pi / 180
def radians_to_degrees(radians):
    return radians * 180 / np.pi

def rotate_vect(vect,degr_=90):

    radians = degrees_to_radians(degr_)
    
    rotation_m = np.array([
        [np.cos(radians),-np.sin(radians)],
        [np.sin(radians),np.cos(radians)]
    ])
    return np.dot(rotation_m,vect)
def rotate_line():
    w = 500
    h = 500
    name = 'rotate line'
    img = np.zeros(shape=(w,h))
    img[:] = 255
    start_p = [int(w/2),int(h/2)]
    cv2.namedWindow(name)
    end_p = [w,int(h / 2)]
   
    while(True):
        cv2.line(img,tuple(start_p),tuple(end_p),(0,255,0),3)
        cv2.imshow(name,img)
        key = cv2.waitKey(500)
        if key == ord('q'):
            break
        elif key == ord('r'):
          img[:] = 255      
          vect_ = np.array(end_p)         
          end_p = list( ( rotate_vect(vect_,15)  ).astype(np.int32) )    
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
   
   A = np.array([[3,1],
                [1,3]])
   vects = np.array([[1,1],[1,-1]])
   vls = np.array([4,2])
   matrices = projections_matrices(A,vls,vects)
   #print(matrices)
   print(matrices[0] + matrices[1])
    
   B = np.array([[9,12],
                 [12,16]])
   vects = np.array([[4/3,-1],
                     [1,4/3]])
   vls = np.array([0,25])
   matrices = projections_matrices(B,vls,vects)
   #print(matrices)
   print(matrices[0] + matrices[1])
   
