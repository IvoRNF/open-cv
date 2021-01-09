import cv2
import numpy as np
import os
from file_loader import FileLoader
from feature import getDescriptor,hot_encode_vect
from util import middleRects

model_fname = './anns/my_ann43.xml'
class MyAnn:

   def __init__(self,input_layer_size=2,output_layer_size=2,hidden_nodes_size=[2],epochs=1,ann_fname='./anns/ann.xml'):
      self.ann_fname = ann_fname
      self.training_data = []
      self.training_labels = []
      self.hidden_nodes_size = hidden_nodes_size
      self.epochs=epochs
      self.input_layer_size = input_layer_size
      self.output_layer_size = output_layer_size
      self.loaded = False
      if(os.path.exists(self.ann_fname)):
        self.ann = cv2.ml.ANN_MLP_load(self.ann_fname)
        self.loaded = True
        print('loaded ann from file')
      else:
         self.createAnn()
   def createAnn(self):
      self.ann = cv2.ml.ANN_MLP_create()
      self.ann.setLayerSizes(np.array([self.input_layer_size,*self.hidden_nodes_size,self.output_layer_size]))
      self.ann.setActivationFunction(cv2.ml.ANN_MLP_BACKPROP,0.1,0.1)
      #self.ann.setTrainMethod(cv2.ml.ANN_MLP_RPROP)
      #self.ann.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM)
      self.ann.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS,100,1.0))
   def fit(self,x,y):
      self.training_labels = y 
      self.training_data = x 
      if not self.loaded:
        self.train()
  
   
   def predict(self,sample : np.ndarray):
      return self.ann.predict(np.array([sample],dtype=np.float32))
         
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

def realtime_teste():
   global model_fname
   ann = MyAnn(input_layer_size=256,hidden_nodes_size=[256//4],output_layer_size=3,
                epochs=1000,ann_fname=model_fname)
   class_names = ['fermento', 'leite_caixa', 'leite_lata']             
   capture = cv2.VideoCapture(0)
   success,frame = capture.read()
   center_pt = (frame.shape[1]//2,frame.shape[0]//2)
   x_center,y_center = center_pt
   curr_rect = None
   i = 0
   for curr_rect in middleRects(frame.shape,center_x=x_center,center_y=y_center):
      if i == 2:
        break
      i+=1

   while (success):
      frame_cpy = frame.copy() 
      cv2.circle(frame_cpy,(x_center,y_center),10,(0,255,255),1)      
      k = cv2.waitKey(5)
      if k == ord('q'):
          break
      success,frame = capture.read()
      x,y,w,h = curr_rect
      cv2.rectangle(frame_cpy,(x,y),(x+w,y+h),(0,255,0),2)
      roi = frame[y:y+h,x:x+w]
      descr = getDescriptor(roi)
      _,stats = ann.predict(descr)
      stats = np.squeeze(stats)

      txt = ('label %s with %.2f' % (class_names[0],stats[0]))
      cv2.putText(frame_cpy,txt,(25,25),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),1)
      
      txt = ('label %s with %.2f' % (class_names[1],stats[1]))
      cv2.putText(frame_cpy,txt,(25,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),1)
      
      txt = ('label %s with %.2f' % (class_names[2],stats[2]))
      cv2.putText(frame_cpy,txt,(25,75),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)

      txt = 'soma %.2f' % (np.sum(stats))
      cv2.putText(frame_cpy,txt,(25,100),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,255),1)
      
      cv2.imshow('',frame_cpy)  
   capture.release()   
   cv2.destroyAllWindows()  

def evaluate_model():
    global model_fname
    loader = FileLoader(dir_to_walk=r'C:\Users\Ivo Ribeiro\Documents\open-cv\datasets\captures')
    loader.load_files()
    x = []
    y = []
    x_test = []
    y_test = []
    
    for row in loader.files: 
       for fname in row['imgs_per_class']:
          img = cv2.imread(fname)
          descr = getDescriptor(img) 
          x.append(descr)
          y.append( hot_encode_vect(len(loader.files),row['index']) )
    for row in loader.files_test: 
       for fname in row['imgs_per_class']:
          img = cv2.imread(fname)
          descr = getDescriptor(img) 
          x_test.append(descr)
          y_test.append( hot_encode_vect(len(loader.files_test),row['index']) )      
    x = np.array(x) 
    y = np.array(y)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    
    print('descriptors loaded.') 
    print(loader.class_names)
    ann = MyAnn(input_layer_size=x.shape[1],hidden_nodes_size=[x.shape[1]//4],output_layer_size=3,
                epochs=1000,ann_fname=model_fname) 
    ann.fit(x=x,y=y)
    
    print('finished train or load.')
    print('evaluating')

    corrects=0   
    for i in np.arange(x_test.shape[0]):
        x = x_test[i]
        y = np.argmax(y_test[i])
        p,stats  = ann.predict(x)
        p = int(p)
        y = int(y)
        stats = np.squeeze(stats)
        print('%s predicted as %s with %.2f' % (loader.class_names[y],loader.class_names[p],stats[p]))
        print(stats)
        if p==y:
           corrects +=1
    print('acc %.2f%s' % ( (corrects/x_test.shape[0]) * 100,'%' ))
   
if __name__ == '__main__':
   print('1 para evaluate\n2 para realtime teste')
   v = input()
   if v=='1':
      evaluate_model()
   elif v=='2':
      realtime_teste()   

    