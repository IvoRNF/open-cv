import cv2
import numpy as np
import os

class TraineerAnn:

   def __init__(self):
      self.ann_fname = './my_ann.xml'
      self.files = []
      self.dirToWalk = r'C:\Users\Ivo Ribeiro\Documents\open-cv\datasets\meus_produtos'
      self.training_data = []
      self.training_labels = []
      self.hidden_nodes_size = 50
      self.default_img_width = 150
      self.default_img_height = 200
      self.input_layer_size = self.default_img_width * self.default_img_height
      self.loaded = False
      if(os.path.exists(self.ann_fname)):
        self.ann = cv2.ml.ANN_MLP_load(self.ann_fname)
        self.loaded = True
        print('loaded ann from file')
      else:
         self.createAnn()
   def createAnn(self):
      self.ann = cv2.ml.ANN_MLP_create()
      self.ann.setLayerSizes(np.array([self.input_layer_size,self.hidden_nodes_size,2]))
      self.ann.setActivationFunction(cv2.ml.ANN_MLP_BACKPROP,0.1,0.1)
      self.ann.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS,100,1.0))
   def run(self):
      if not self.loaded:
        self.load_files() 
        self.train()
   def load_files(self):          
      i = 0
      for root,dirs,files in os.walk(self.dirToWalk):
          for name in files:
            basename = os.path.basename(root)
            y = self.files_contains_class_name_index_of(basename)
            if (y == -1):
              y = len(self.files)   
              self.files.append({"index":i,"imgs_per_class":[],"class_name":basename})
              i += 1 
            row = self.files[y]
            files_per_class = row["imgs_per_class"]
            files_per_class.append(os.path.join(root,name))   
   
   def files_contains_class_name_index_of(self, name : str):
       i = 0
       for f in self.files:
           row = self.files[i]
           if name == row['class_name']:
               return i
           i += 1    
       return -1        
   
   def predict(self,img : np.ndarray):
      sample = img.copy()
      shape = (self.default_img_height,self.default_img_width)
      if sample.shape != (self.input_layer_size,):
         if sample.shape != shape:
             sample = cv2.resize(sample,shape,interpolation=cv2.INTER_LINEAR)
         sample = sample.reshape(self.input_layer_size)
      return self.ann.predict(np.array([sample],dtype=np.float32))   
   def train(self,epochs=3):
       classes = {
           0:[0,0],
           1:[0,1]
       }
       for epoch in range(epochs):
          for row in self.files:
            for file_name in row['imgs_per_class']:
               print('training %d ...' % (epoch))
               img = cv2.imread(file_name,cv2.IMREAD_GRAYSCALE)
               img = img.astype(np.float32)
               class_idx = row['index']
               response = np.array([classes[class_idx]],dtype=np.float32)
               img = img.reshape(img.size)
               img = np.array([img],dtype=np.float32)
               data = cv2.ml.TrainData_create(img,cv2.ml.ROW_SAMPLE,response)
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

 
   
   def get_class_name(self,index : int):
        return self.files[index]['class_name']

  

      
          
if __name__ == '__main__':
   ann = TraineerAnn()    
   ann.run()
   files_to_test = [
      r'C:\Users\Ivo Ribeiro\Documents\open-cv\datasets\meus_produtos\leite_po\IMG_20200914_080715874_BURST008.jpg',
       r'C:\Users\Ivo Ribeiro\Documents\open-cv\datasets\meus_produtos\creme_leite_\IMG_20200829_094657.jpg',
      r'C:\Users\Ivo Ribeiro\Documents\open-cv\datasets\originals_excluded\fermento\IMG_20200829_094931.jpg'
   ]
   for fname in files_to_test:
     img = cv2.imread(fname,cv2.IMREAD_GRAYSCALE)
     dname = os.path.dirname(fname)
     base = os.path.basename(dname)
     print( 'arquivo da pasta %s \n' % (base) )
     print(ann.predict(img))



