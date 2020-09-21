import cv2
import numpy as np
import os

class Knn:
    def __init__(self):
        self.knn_fname = './my_knn.xml'
        if os.path.exists(self.knn_fname):
          print('loading knn from file')  
          self.knn = cv2.ml.KNearest_load(self.knn_fname)  
        else:
          print('savind knn to file')  
          self.knn = cv2.ml.KNearest_create()  
    def train(self,data , labels):
        self.knn.train(data,cv2.ml.ROW_SAMPLE,labels)
        self.knn.save(self.knn_fname)
    def predict(self,sample , k = 3):
        return self.knn.findNearest(sample, k)
 

if __name__ == '__main__':
    knn = Knn()

    test_samples_class1 = np.random.randint(0,2,(8,6)).astype(np.float32)
    test_samples_class2 = np.random.randint(3,8,(8,6)).astype(np.float32)
    samples = np.vstack((test_samples_class1,test_samples_class2))
    responses = np.array([1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2],dtype=np.float32)
    knn.train(samples,responses)
    ret, results, neighbours, dist = knn.predict(np.array([[5,5,5,5,5,5]],dtype=np.float32),3)
    print("result: {}".format(results))
    print("neighbours: {}".format(neighbours))
    print("distance: {}".format(dist))
    print('ret {}'.format(ret))
