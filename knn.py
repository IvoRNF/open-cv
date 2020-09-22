import cv2
import numpy as np
import os

class Knn:
    
    def __init__(self):
        self.files = []
        self.dir_to_walk = r'C:\Users\Ivo Ribeiro\Documents\open-cv\datasets\captures'
        self.knn_fname = './my_knnB.xml'
        self.loaded = False
        self.shape = (100,64)
        self.hog = self.createHog()
        if os.path.exists(self.knn_fname):
            print('loading knn from file')
            self.loaded = True
            self.knn = cv2.ml.KNearest_load(self.knn_fname)  
        else:
            print('saving knn to file')  
            self.knn = cv2.ml.KNearest_create()
    def run(self):
        if not self.loaded:
          self.load_files()
          self.train_knn()

    def train_knn(self):

         responses = []
         descriptors = []
         for row in self.files: 
             dirs = row['imgs_per_class']
             idx = row['index']
             for file_name in dirs:
                img = cv2.imread(file_name,cv2.IMREAD_GRAYSCALE)
                self.tryReshape(img)
                descriptor = self.hog.compute(img)
                descriptors.append(descriptor)
                responses.append(idx)
         print('training')       
         knn.train(np.array(descriptors,dtype=np.float32),np.array(responses,dtype=np.float32))
         print('trained')
        
    def createHog(self):
        hog = cv2.HOGDescriptor(self.shape,(8,8),(4,4),(8,8),9,1,-1,0,0.2,1,64,True)
        return hog
    def load_files(self):          
      i = 0
      for root,dirs,files in os.walk(self.dir_to_walk):
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
    def train(self,data , labels):
        self.knn.train(data,cv2.ml.ROW_SAMPLE,labels)
        self.knn.save(self.knn_fname)
    def predict(self,sample , k = 3):
        return self.knn.findNearest(sample, k)
    def tryReshape(self,img):
        if img.shape != self.shape:
            w = self.shape[1]
            h = self.shape[0]
            img = cv2.resize(img,(w,h),interpolation=cv2.INTER_AREA)

if __name__ == '__main__':
    knn = Knn()
    knn.run()
    files_to_test = [
            r'C:\Users\Ivo Ribeiro\Documents\open-cv\datasets\captures\leite_caixa\1.jpg',
            r'C:\Users\Ivo Ribeiro\Documents\open-cv\datasets\captures\leite_lata\181.jpg'  ,
            r'C:\Users\Ivo Ribeiro\Documents\open-cv\datasets\captures\leite_lata\350.jpg',
            r'C:\Users\Ivo Ribeiro\Documents\open-cv\datasets\meus_produtos\creme_leite_\IMG_20200829_094657.jpg',
            r'C:\Users\Ivo Ribeiro\Documents\open-cv\datasets\meus_produtos\leite_po\IMG_20200910_125946964_BURST000_COVER.jpg',
            r'C:\Users\Ivo Ribeiro\Documents\open-cv\datasets\originals\leite_po\IMG_20200910_125946964_BURST003.jpg'
            
    ]
    descriptors = []
    for fname in files_to_test:
        img = cv2.imread(fname,cv2.IMREAD_GRAYSCALE)
        knn.tryReshape(img)
        descriptor = knn.hog.compute(img)
        descriptors.append(descriptor)

    ret, results, neighbours, dist = knn.predict(np.array(descriptors,dtype=np.float32))
    print( 'results %s '% (results) )
    print( 'ret %s' % (ret) )
    print( 'dist %s' % (dist) )

