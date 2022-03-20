import numpy as np
import cv2

''' solving diferential equations MIT 1806 course
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
    print(start_p)
    cv2.namedWindow(name)
    end_p = [w,int(h / 2)]
    degr = 90    
    while(True):
         
        cv2.line(img,tuple(start_p),tuple(end_p),(0,255,0),4)
        cv2.imshow(name,img)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break
        elif key == ord('r'):
          img[:] = 255
          degr += 10
          vect_ = np.array(end_p) - np.array(start_p)
          len_ = np.sqrt( np.dot(vect_.T,vect_) )
          vect_ = vect_ / len_
          end_p = list( ((rotate_vect(vect_,degr) * len_) + np.array(start_p)).astype(np.int32) )
          if degr==360:
             degr = 0
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
   rotate_line()
    
