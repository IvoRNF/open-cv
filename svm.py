import cv2 
import os 
from file_loader import FileLoader

class Svm(FileLoader):

    def __init__(self, C=2,G=0.5):
        super().__init__(dir_to_walk= r'C:\Users\Ivo Ribeiro\Documents\open-cv\datasets\captures')
        self.shape = (200,100)
        self.hog = self.createHog(self.shape)
        self.svm_fname = 'my_svm.xml'
        if os.path.exists(self.svm_fname):
            print('loading svm from file.')
            self.svm = cv2.ml.SVM_load(self.svm_fname)
        else:
            print('creating svm model.')
            self.svm = cv2.ml.SVM_create()
            self.svm.setType(cv2.ml.SVM_C_SVC)
            self.svm.setKernel(cv2.ml.SVM_RBF)
            self.svm.setC(C)
            self.svm.setGamma(G)
    def trainAndSave(self): 
        self.svm.train(np.array([]), cv2.ml.ROW_SAMPLE,np.array([]))
        self.svm.save(self.svm_fname)
    def createHog(self,shape):
        #necessario aspect raio 1:2
        w_h = shape[::-1]
        hog = cv2.HOGDescriptor(w_h,(8,8),(4,4),(8,8),9,1,-1,0,0.2,1,64,True)
        return hog    
    def getHogDescriptor(self,sample):
      sampleToPredict = sample
      if len(sampleToPredict.shape)>2:
        sampleToPredict = remove_ilumination(sampleToPredict)
        sampleToPredict = cv2.cvtColor(sampleToPredict,cv2.COLOR_BGR2GRAY)  
      if sampleToPredict.shape != self.shape:
          reversedShape = self.shape[::-1]
          sampleToPredict = cv2.resize(sampleToPredict,reversedShape,interpolation=cv2.INTER_AREA)  
      descr = self.hog.compute(sampleToPredict)
      descr = np.squeeze(descr)