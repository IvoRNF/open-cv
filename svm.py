import cv2 
import os 
from file_loader import FileLoader
import numpy as np

class Svm(FileLoader):

    def __init__(self, C=16,G=0.5):
        super().__init__(dir_to_walk= r'C:\Users\Ivo Ribeiro\Documents\open-cv\datasets\captures')
        self.shape = (200,100)
        self.hog = self.createHog(self.shape)
        self.svm_fname = './my_svm.xml'
        if os.path.exists(self.svm_fname):
            self.loaded = True
            print('loading svm from file.')
            self.svm = cv2.ml.SVM_load(self.svm_fname)
        else:
            self.loaded = False
            print('creating svm model.')
            self.svm = cv2.ml.SVM_create()
            self.svm.setType(cv2.ml.SVM_C_SVC)
            self.svm.setKernel(cv2.ml.SVM_RBF)
            self.svm.setC(C)
            self.svm.setGamma(G)
            self.trainAndSave()
    def trainAndSave(self): 
        self.load_files()
        descriptors = []
        responses = []
        for row in self.files:
           dirs = row['imgs_per_class']
           idx = row['index']
           for file_name in dirs:
              img = cv2.imread(file_name) 
              descriptor = self.getHogDescriptor( img )
              descriptors.append(descriptor)
              responses.append(idx) 
        descriptors = np.array(descriptors)      
        responses = np.array(responses)
        self.svm.train(descriptors, cv2.ml.ROW_SAMPLE,responses)
        self.svm.save(self.svm_fname)
        print('file saved.')
    def createHog(self,shape):
        #necessario aspect raio 1:2
        w_h = shape[::-1]
        hog = cv2.HOGDescriptor(w_h,(8,8),(4,4),(8,8),9,1,-1,0,0.2,1,64,True)
        return hog    
    def getHogDescriptor(self,sample):
      sampleToPredict = sample
      if len(sampleToPredict.shape)>2:
        sampleToPredict = cv2.cvtColor(sampleToPredict,cv2.COLOR_BGR2GRAY)  
      if sampleToPredict.shape != self.shape:
          reversedShape = self.shape[::-1]
          sampleToPredict = cv2.resize(sampleToPredict,reversedShape,interpolation=cv2.INTER_AREA)  
      descr = self.hog.compute(sampleToPredict)
      descr = np.squeeze(descr)
      return descr


def evaluate_svm():
    svm = Svm()
    if svm.loaded:
       svm.load_files() 
    eval_arr = np.zeros(len(svm.files_test),dtype=np.int8)  
    for row in svm.files_test:
        class_idx = row['index']
        class_name = row['class_name']
        for fname in row['imgs_per_class']:
          img = cv2.imread(fname)
          descr = svm.getHogDescriptor(img)
          scores = None
          response = svm.svm.predict(np.array([descr]))
          predicted_class_idx = response[1][0][0]
          correct = (predicted_class_idx==class_idx)
          if(correct):
             eval_arr[class_idx] += 1
          print('%s %d classificado como %d - arquivo %s'
                % (class_name,class_idx,predicted_class_idx, os.path.basename(fname) ))
        for i in range( eval_arr.shape[0] ): 
            count_corrects = eval_arr[i]
            row = svm.files_test[i]
            count_per_class = len(row['imgs_per_class'])
            class_name = row['class_name']
            print('acurracy %.2f%s para %s' % ( ((count_corrects/count_per_class) * 100),'%',class_name ))         

if __name__ == '__main__':
    evaluate_svm()
