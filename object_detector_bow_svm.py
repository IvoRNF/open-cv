import cv2
import numpy as np
import os

class Traineer:

   def __init__(self):    
      self.sift = cv2.xfeatures2d.SIFT_create()
      self.FLANN_INDEX_KDTREE = 1
      self.flann = cv2.FlannBasedMatcher(
        dict(algorithm=self.FLANN_INDEX_KDTREE,trees=5),{})
      num_clusters = 131
      self.bow_kmeans_trainer = cv2.BOWKMeansTrainer( num_clusters )  
      self.bow_extractor = cv2.BOWImgDescriptorExtractor(self.sift,self.flann)
      self.files = []
      self.dirToWalk = r'.\datasets\5857_1166105_bundle_archive\fruits-360\Test'
      self.training_data = []
      self.training_labels = []
      self.svm = None
   def files_contains_class_name_index_of(self, name : str):
       i = 0
       for f in self.files:
           row = self.files[i]
           if name == row['class_name']:
               return i
           i += 1    
       return -1        
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

   def fill_bow(self):
       for row in self.files:
          for file_name in row['imgs_per_class'][0:15]: 
             img = cv2.imread(file_name,cv2.IMREAD_GRAYSCALE)
             keypoints, descriptors = self.sift.detectAndCompute(img, None)
             if descriptors is not None:
                self.bow_kmeans_trainer.add(descriptors)
       voc = self.bow_kmeans_trainer.cluster()
       self.bow_extractor.setVocabulary(voc)          
   def extract_bow_descriptors(self,img : np.ndarray):
       features = self.sift.detect(img)
       return self.bow_extractor.compute(img, features)
   def separate_training_data(self):
     for row in self.files:
          for file_name in row['imgs_per_class'][0:15]:
             img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
             descriptors = self.extract_bow_descriptors(img)
             if descriptors is not None:
                self.training_data.extend(descriptors)
                self.training_labels.append(row['index'])
   def train(self):
      self.svm = cv2.ml.SVM_create()
      '''if(os.path.exists('./fruits_svm.xml')):
         self.svm.load('./fruits_svm.xml')
         print('loaded SVM from file')
      else:  ''' 
      self.svm.train(np.array(self.training_data), cv2.ml.ROW_SAMPLE,
            np.array(self.training_labels))
      #self.svm.save('./fruits_svm.xml')
      #print('saved SVM to file')
   def run(self):
      self.load_files()
      self.fill_bow()
      self.separate_training_data()
      self.train()
   def get_label_name(self,index : int):
        return self.files[index]['class_name']
      
   def test(self):
      img = cv2.imread('./carambola.jpg')
      gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      descriptors = self.extract_bow_descriptors(gray_img)
      prediction = self.svm.predict(descriptors)
      label_idx = int(prediction[1][0][0])
      print( self.get_label_name(label_idx) )
if __name__ == '__main__':
   trainner = Traineer()    
   trainner.run()
   trainner.test()  
''' 

    
     
for test_img_path in ['CarData/TestImages/test-0.pgm',
     'CarData/TestImages/test-1.pgm',
     '../images/car.jpg',
     '../images/haying.jpg',
     '../images/statue.jpg',
     '../images/woodcutters.jpg']:
     img = cv2.imread(test_img_path)
     gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
     descriptors = extract_bow_descriptors(gray_img)
     prediction = svm.predict(descriptors)
     if prediction[1][0][0] == 1.0:
         text = 'car'
         color = (0, 255, 0)
     else:
         text = 'not car'
         color = (0, 0, 255)
     cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
         color, 2, cv2.LINE_AA)
     cv2.imshow(test_img_path, img)
cv2.waitKey(0)     
'''     
     




