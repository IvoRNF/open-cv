import cv2
import numpy as np
import os

class Traineer:

   def __init__(self):    
      self.sift = cv2.xfeatures2d.SIFT_create()
      self.FLANN_INDEX_KDTREE = 1
      self.flann = cv2.FlannBasedMatcher(
        dict(algorithm=self.FLANN_INDEX_KDTREE,trees=5),{})
      num_clusters = 40
      self.bow_kmeans_trainer = cv2.BOWKMeansTrainer( num_clusters )  
      self.bow_extractor = cv2.BOWImgDescriptorExtractor(self.sift,self.flann)
      self.files = []
      self.dirToWalk = r'.\datasets\5857_1166105_bundle_archive\fruits-360\Test'   
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
              i += 1   
              self.files.append({"index":i,"dirs_per_class":[],"class_name":basename})
            row = self.files[y]
            files_per_class = row["dirs_per_class"]
            files_per_class.append(name)

   def fill_bow(self):
       print( len(self.files) )             
if __name__ == '__main__':
   trainner = Traineer()    
   trainner.load_files() 
   trainner.fill_bow()   
     
'''

def add_sample(path):
 img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
 keypoints, descriptors = sift.detectAndCompute(img, None)
 if descriptors is not None:
    bow_kmeans_trainer.add(descriptors)


for i in range(BOW_NUM_TRAINING_SAMPLES_PER_CLASS):
 pos_path, neg_path = get_pos_and_neg_paths(i)
 add_sample(pos_path)
 add_sample(neg_path)
 
voc = bow_kmeans_trainer.cluster()
bow_extractor.setVocabulary(voc) 


def get_pos_and_neg_paths(i):
 pos_path = 'CarData/TrainImages/pos-%d.pgm' % (i+1)
 neg_path = 'CarData/TrainImages/neg-%d.pgm' % (i+1)
 return pos_path, neg_path
 
def extract_bow_descriptors(img):
 features = sift.detect(img)
 return bow_extractor.compute(img, features)
 

training_data = []
training_labels = []
for i in range(SVM_NUM_TRAINING_SAMPLES_PER_CLASS):
 pos_path, neg_path = get_pos_and_neg_paths(i)
 pos_img = cv2.imread(pos_path, cv2.IMREAD_GRAYSCALE)
 pos_descriptors = extract_bow_descriptors(pos_img)
 if pos_descriptors is not None:
     training_data.extend(pos_descriptors)
     training_labels.append(1)
     neg_img = cv2.imread(neg_path, cv2.IMREAD_GRAYSCALE)
     neg_descriptors = extract_bow_descriptors(neg_img)
 if neg_descriptors is not None:
     training_data.extend(neg_descriptors)
     training_labels.append(-1)

svm = cv2.ml.SVM_create()
svm.train(np.array(training_data), cv2.ml.ROW_SAMPLE,
            np.array(training_labels))     
     
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
     




