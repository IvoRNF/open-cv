import cv2
import numpy as np
import os

class Knn:
    
    def __init__(self):
        self.files = []
        self.files_test = []
        self.PERC_TO_TEST = 0.2  #20 por cento para teste  
        self.dir_to_walk = r'C:\Users\Ivo Ribeiro\Documents\open-cv\datasets\captures'
        self.knn_fname = './my_knnB.xml'
        self.loaded = False
        self.shape = (100,75)
        self.hog = self.createHog()
        self.class_names = os.listdir(self.dir_to_walk)
        if os.path.exists(self.knn_fname):
            print('loading knn from file')
            self.loaded = True
            self.knn = cv2.ml.KNearest_load(self.knn_fname)
            print('loaded') 
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
                img = self.pyrDown(img)
                descriptor = self.hog.compute(img)
                descriptors.append(descriptor)
                responses.append(idx)
         print('training')
         descriptors = np.squeeze(descriptors)
         self.train(
                        np.array(descriptors,dtype=np.float32),
                        np.array(responses,dtype=np.float32)
                        )
         print('trained')
        
    def createHog(self):
        hog = cv2.HOGDescriptor((64,64),(8,8),(4,4),(8,8),9,1,-1,0,0.2,1,64,True)
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
      for row in self.files:
          imgs = row['imgs_per_class']
          np.random.shuffle(imgs)
          len_test = int(len(imgs)*self.PERC_TO_TEST)
          imgs_test ,row['imgs_per_class'] = np.split(imgs,[len_test])
          self.files_test.append({"index":row['index'],"imgs_per_class":imgs_test,"class_name":row['class_name']})
      
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
    def processAndPredict(self,sample , k = 20,pyrDownLevels=0):
        sampleToPredict = sample
        if type(sampleToPredict) != np.ndarray:
            sampleToPredict = np.array(sampleToPredict,dtype=np.float32)
        if len(sampleToPredict.shape)>2:
          sampleToPredict = cv2.cvtColor(sampleToPredict,cv2.COLOR_BGR2GRAY)
        if sampleToPredict.shape != self.shape:
           reversedShape = self.shape[::-1]
           sampleToPredict = cv2.resize(sampleToPredict,reversedShape,interpolation=cv2.INTER_AREA)     
        sampleToPredict = self.pyrDown(sampleToPredict,pyrDownLevels)
        descriptor = self.hog.compute(sampleToPredict)     
        return self.knn.findNearest(np.array([descriptor],dtype=np.float32), k)
    def pyrDown(self,img,levels=1):
        for i in range(levels):
            img = cv2.pyrDown(img)
        return img
def real_time_test():
   min_distance = 20000.00
   knn = Knn()
   knn.run()
   capture = cv2.VideoCapture(0)
   sucess,frame = capture.read()
   roi_width = int(frame.shape[1] * 0.5)
   roi_height = int(frame.shape[0] * 0.5)
   middle_w = int(frame.shape[1]/2)
   middle_h = int(frame.shape[0]/2)
   x = middle_w - int(roi_width / 2)
   y = middle_h - int(roi_height / 2)
   while (sucess):
      frame_cpy = frame.copy() 
      cv2.rectangle(frame_cpy,(x,y),(x+roi_height,y+roi_width),(0,255,0),1) #inverte em paisagem
      cv2.imshow('',frame_cpy)
      frame_cpy = None
      k = cv2.waitKey(30)
      if k == ord('f'):
          break
      gap = 10 
      roi : np.ndarray = frame[y:y+roi_width-gap,x:x+roi_height-gap] 
      response = knn.processAndPredict(roi)
      distance = np.sum ( np.squeeze(response[3]) )
      class_idx = int(response[0])
      sucess,frame = capture.read()
      if(distance < min_distance):
          class_name = knn.class_names[class_idx]
          txt = '%s(%.2f)' % (class_name,distance)
          cv2.putText(frame,txt,(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1,cv2.LINE_AA)
          #print(class_name,end='\n\n')
       
   capture.release()   
   cv2.destroyAllWindows()    
 
def evaluate_knn():
    knn = Knn()
    knn.run()
    if knn.loaded: #quando carrega do xml precisa carregar os arquivos para teste
      knn.load_files()
    eval_arr = np.zeros(len(knn.files_test),dtype=np.int8)  
    for row in knn.files_test:
        class_idx = row['index']
        class_name = row['class_name']
        for fname in row['imgs_per_class']:
          img = cv2.imread(fname,cv2.IMREAD_GRAYSCALE)
          response = knn.processAndPredict(img)
          predicted_class_idx = response[0]
          correct = (predicted_class_idx==class_idx)
          if(correct):
             eval_arr[class_idx] += 1
          print('%s %d classificado como %d - arquivo %s'
                % (class_name,class_idx,predicted_class_idx, os.path.basename(fname) ))
    for i in range( eval_arr.shape[0] ): 
       count_corrects = eval_arr[i]
       row = knn.files_test[i]
       count_per_class = len(row['imgs_per_class'])
       class_name = row['class_name']
       print('acurracy %d%s para %s' % ( ((count_corrects//count_per_class) * 100),'%',class_name ))     

def capture():
   capture = cv2.VideoCapture(0)
   sucess,frame = capture.read()
   roi_width = int(frame.shape[1] * 0.5)
   roi_height = int(frame.shape[0] * 0.5)
   middle_w = int(frame.shape[1]/2)
   middle_h = int(frame.shape[0]/2)
   x = middle_w - int(roi_width / 2)
   y = middle_h - int(roi_height / 2)
   i = 1000
   while (sucess):
     frame_cpy = frame.copy() 
     cv2.rectangle(frame_cpy,(x,y),(x+roi_height,y+roi_width),(0,255,0),1) #inverte em paisagem
     cv2.imshow('',frame_cpy)
     frame_cpy = None
     k = cv2.waitKey(30)
     if k == ord('f'):
        break
     gap = 10 
     if k == ord('l'): #salva foto de leite lata
        i = i + 1
        roi = frame[y:y+roi_width-gap,x:x+roi_height-gap]
        roi = cv2.resize(roi,(150,200),interpolation=cv2.INTER_AREA)
        cv2.imwrite('./datasets/captures/leite_lata/%d.jpg'%(i),roi)
     elif k == ord('c'): #salva foto de leite caixa
        i = i + 1
        roi = frame[y:y+roi_width-gap,x:x+roi_height-gap]
        roi = cv2.resize(roi,(150,200),interpolation=cv2.INTER_AREA)
        cv2.imwrite('./datasets/captures/leite_caixa/%d.jpg'%(i),roi)   
     sucess,frame = capture.read()     
   capture.release()   
   cv2.destroyAllWindows()


if __name__ == '__main__':
    print('1 para evaluate \n2 para real time test\n3 capturar ')
    v = input()
    if v =='1':  
      evaluate_knn()
    elif v=='2':
      real_time_test()
    elif v=='3':
      capture()  

 
