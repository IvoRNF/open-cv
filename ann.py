import cv2
import numpy as np
import os

class TraineerAnn:

   def __init__(self):
      self.ann_fname = './anns/my_ann222.xml'
      self.training_data = []
      self.training_labels = []
      self.hidden_nodes_size = 2
      self.epochs=1
      self.input_layer_size = 2
      self.output_layer_size = 2
      self.loaded = False
      if(os.path.exists(self.ann_fname)):
        self.ann = cv2.ml.ANN_MLP_load(self.ann_fname)
        self.loaded = True
        print('loaded ann from file')
      else:
         self.createAnn()
   def createAnn(self):
      self.ann = cv2.ml.ANN_MLP_create()
      self.ann.setLayerSizes(np.array([self.input_layer_size,self.hidden_nodes_size,self.output_layer_size]))
      self.ann.setActivationFunction(cv2.ml.ANN_MLP_BACKPROP,0.1,0.1)
      #self.ann.setTrainMethod(cv2.ml.ANN_MLP_RPROP)
      self.ann.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM)
      self.ann.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS,100,1.0))
   def fit(self,x,y):
      self.training_labels = y 
      self.training_data = x 
      if not self.loaded:
        self.train()
  
   
   def predict(self,sample : np.ndarray):
      return self.ann.predict(np.array([sample],dtype=np.float32))
   def vect_class_idx(self,class_idx):
      arr = []
      for _ in range(self.output_layer_size):
         arr.append(0)
      arr[class_idx] = class_idx   
      return arr  
         
   def train(self):
       train_labels = np.array(self.training_labels,dtype=np.float32)
       train_data = np.array(self.training_data,dtype=np.float32)
       for _ in range(self.epochs):     
         data = cv2.ml.TrainData_create(train_data,cv2.ml.ROW_SAMPLE,train_labels)
         if self.ann.isTrained():
               self.ann.train(data,
                     cv2.ml.ANN_MLP_UPDATE_WEIGHTS |
                     cv2.ml.ANN_MLP_NO_INPUT_SCALE |
                     cv2.ml.ANN_MLP_NO_OUTPUT_SCALE
                     )
         else:
               self.ann.train(data,
                     cv2.ml.ANN_MLP_NO_INPUT_SCALE |
                     cv2.ml.ANN_MLP_NO_OUTPUT_SCALE
                     )
       self.ann.save(self.ann_fname)                   
       print('train complete')    

 

   
if __name__ == '__main__':
    ann = TraineerAnn() 
    ann.epochs = 1000
    x = [[0,255],[255,0],[0,255],[0,255]]
    y = [[1,0],[0,1],[1,0],[1,0]]
    ann.fit(x=x,y=y)
    

    test = x
    expected = [np.argmax(z) for z in y]
    for sample,target in zip(test,expected): 
       label,propab = ann.predict(sample)
       if label != target:
         print(propab)
         print(sample) 
         print(label)
         print(target)
         print('...')
    print('ok')    
    ''' 
    for i in range(10): 
      try:
       w = ann.ann.getWeights(i)
       print(w)
      except:
         break
    print(ann.ann.getLayerSizes())  
    '''
       
