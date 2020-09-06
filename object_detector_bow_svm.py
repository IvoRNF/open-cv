import cv2
import numpy as np
import os

class Traineer:

   def __init__(self):
      self.is_test = os.path.exists('./my_svm.xml')
      self.sift = cv2.xfeatures2d.SIFT_create()
      self.FLANN_INDEX_KDTREE = 1
      self.SVM_SCORE_THRESHOLD = 3
      self.flann = cv2.FlannBasedMatcher(
        dict(algorithm=self.FLANN_INDEX_KDTREE,trees=5),{})
      num_clusters = 12
      self.bow_kmeans_trainer = cv2.BOWKMeansTrainer( num_clusters )  
      self.bow_extractor = cv2.BOWImgDescriptorExtractor(self.sift,self.flann)
      self.files = []
      self.dirToWalk = r'C:\Users\Ivo Ribeiro\Documents\open-cv\datasets\meus_produtos'
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
          for file_name in row['imgs_per_class']: 
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
     if self.is_test:
        return
     for row in self.files:
          for file_name in row['imgs_per_class']:
             img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
             descriptors = self.extract_bow_descriptors(img)
             if descriptors is not None:
                self.training_data.extend(descriptors)
                self.training_labels.append(row['index'])
   def train_or_load(self):
      if(self.is_test):
         self.svm = cv2.ml.SVM_load('./my_svm.xml')
         print('loaded SVM from file')
      else:
        self.svm = cv2.ml.SVM_create()
        self.svm.setType(cv2.ml.SVM_C_SVC)
        self.svm.setC(50)
        self.svm.setGamma(0.5)
        self.svm.setKernel(cv2.ml.SVM_RBF)

        self.svm.train(np.array(self.training_data), cv2.ml.ROW_SAMPLE,
            np.array(self.training_labels))
        self.svm.save('./my_svm.xml')
        print('saved SVM to file')
   def run(self):
      self.load_files()
      self.fill_bow()
      self.separate_training_data()
      self.train_or_load()
   def get_class_name(self,index : int):
        return self.files[index]['class_name']

   def pyramid(self,img, scale_factor=1.25, min_size=(100, 100),max_size=(600, 600)):
       h, w = img.shape
       min_w, min_h = min_size
       max_w, max_h = max_size
       while w >= min_w and h >= min_h:
         if w <= max_w and h <= max_h:
           yield img
         w /= scale_factor
         h /= scale_factor
         img = cv2.resize(img, (int(w), int(h)),interpolation=cv2.INTER_AREA)   
   def test(self):

      img = cv2.imread(r'C:\Users\Ivo Ribeiro\Documents\open-cv\datasets\originals\IMG_20200831_080315.jpg')
      gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

      max_score = -1
      prediction_class_idx = -1
      pyrlevel = 0
      max_score_pyrlevel = -1
      for resized_img in self.pyramid(gray_img,max_size=gray_img.shape):
          pyrlevel = pyrlevel + 1
          descriptors = self.extract_bow_descriptors(resized_img)
          if descriptors is None:
             continue                   
          prediction = self.svm.predict(descriptors)
          class_idx = int(prediction[1][0][0])
          raw_prediction = self.svm.predict(descriptors,flags=cv2.ml.STAT_MODEL_RAW_OUTPUT)
          score = raw_prediction[1][0][0]
          print('score %d at %d level, class %s' % (score,pyrlevel,self.get_class_name(class_idx))) 
          if score > max_score:
              max_score = score
              max_score_pyrlevel = pyrlevel
              prediction_class_idx = class_idx
      if(max_score >= 0):
        print( 'encontrou %s com score %d no level %d da piramide' % (self.get_class_name(prediction_class_idx),max_score,max_score_pyrlevel) )
          
if __name__ == '__main__':
   trainner = Traineer()    
   trainner.run()
   trainner.test()  




