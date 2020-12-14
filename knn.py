import cv2
import numpy as np
import os
import time
import math
import matplotlib.pyplot as plt 
from skimage import exposure
from file_loader import FileLoader
import dlib 
from sklearn.manifold import TSNE
from feature import getDescriptor,pre_process
from util import middleRects

class Knn(FileLoader):
    
    def __init__(self):
        super().__init__(dir_to_walk=r'C:\Users\Ivo Ribeiro\Documents\open-cv\datasets\captures')
        self.knn_fname = './my_knn.xml'
        self.loaded = False
        self.shape = (128,64)
        if os.path.exists(self.knn_fname):
            print('loading knn from file')
            self.loaded = True
            self.knn = cv2.ml.KNearest_load(self.knn_fname)
            print('loaded') 
        else:
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
                descriptor = getDescriptor( cv2.imread(file_name,cv2.IMREAD_GRAYSCALE),self.shape)
                descriptors.append(descriptor)
                responses.append(idx)
         print('training')
         descriptors = np.squeeze(descriptors)
         self.train(
                        np.array(descriptors,dtype=np.float32),
                        np.array(responses,dtype=np.float32)
                        )
         print('trained')
                   
    def train(self,data , labels):
        self.knn.train(data,cv2.ml.ROW_SAMPLE,labels)
        self.knn.save(self.knn_fname)
        print('saving knn to file')  
    
  
    def processAndPredict(self,sample , k = 6):
        descriptor = getDescriptor(sample,self.shape)  
        return self.knn.findNearest(np.array([descriptor],dtype=np.float32), k)
    def pyrDown(self,img,levels=1):
        for _ in range(levels):
            img = cv2.pyrDown(img)
        return img
  
def real_time_test():
   min_distance = 0.01
   knn = Knn()
   knn.run()
   capture = cv2.VideoCapture(0)
   success,frame = capture.read()
   center_pt = (frame.shape[1]//2,frame.shape[0]//2)
   x_center,y_center = center_pt
   curr_rect = None
   i = 0
   not_found_count = 0
   for curr_rect in middleRects(frame.shape,center_x=x_center,center_y=y_center):
      if i == 2:
        break
      i+=1
   tracker = None   
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
      response = knn.processAndPredict(roi)
      distance = np.sum ( np.squeeze(response[3]) )
      #cv2.putText(frame_cpy,'%.2f %s' % (distance,knn.class_names[int(response[0])]),(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2) 
      if(distance <= min_distance):
        tracker = dlib.correlation_tracker()
        t_rect = dlib.rectangle(x,y,x+w,y+h)
        tracker.start_track( cv2.cvtColor(frame,cv2.COLOR_BGR2RGB) ,t_rect)
        cv2.rectangle(frame_cpy,(x,y),(x+w,y+h),(0,255,0),2)  
        class_idx = int(response[0])
        class_name = knn.class_names[class_idx]
        txt = '%s(%.2f)' % (class_name,distance) 
        cv2.putText(frame_cpy,txt,(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2) 
      else: 
        if tracker is not None: 
           tracker.update(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))  
           pos = tracker.get_position()
           frame_cpy = frame.copy()
           roi = frame[int(pos.top()):int(pos.bottom()),int(pos.left()):int(pos.right())]
           if(roi.shape[0]==0)or(roi.shape[1]==0):
              distance = 99999
           else:
              response = knn.processAndPredict(roi)
              distance = np.sum ( np.squeeze(response[3]) )
           if(distance < min_distance):
               not_found_count = 0
               txt = '%s(%.2f)' % (class_name,distance) 
               cv2.putText(frame_cpy,txt,(int(pos.left()),int(pos.top())),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
           else:
             not_found_count += 1
             if not_found_count==150: #para de perseguir depois de 150 frames com roi nao detectado
               not_found_count =0
               tracker = None
           cv2.rectangle(frame_cpy,(int(pos.left()),int(pos.top())),(int(pos.right()),int(pos.bottom())),(0,255,0),2)    
      cv2.imshow('',frame_cpy)  
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
          img = cv2.imread(fname)
          response = knn.processAndPredict(img)
          predicted_class_idx = response[0]
          correct = (predicted_class_idx==class_idx)
          if(correct):
             eval_arr[class_idx] += 1
          distance = np.sum( np.squeeze(response[3]) )
          print( '%s %d classificado como %d - arquivo %s, distance %.2f'
                % (class_name,class_idx,predicted_class_idx, os.path.basename(fname),distance ))
    for i in range( eval_arr.shape[0] ): 
       count_corrects = eval_arr[i]
       row = knn.files_test[i]
       count_per_class = len(row['imgs_per_class'])
       class_name = row['class_name']
       print('acurracy %.2f%s para %s' % ( ((count_corrects/count_per_class) * 100),'%',class_name ))     


def show_std(): 
  knn = Knn()
  knn.run()
  if knn.loaded: 
    knn.load_files()
  descriptors = []
  for row in knn.files:
     imgs = row['imgs_per_class']
     for fname in imgs: 
       descriptors.append(  getDescriptor( cv2.imread(fname , cv2.IMREAD_GRAYSCALE) ),knn.shape  )
  descriptors = np.array(descriptors)
  stds = np.std(descriptors,axis=0)
  stds = stds * 100
  stds = np.int8(stds)
  min_bin = np.min(stds)
  max_bin = np.max(stds)
  bins = (max_bin-min_bin)
  hist = np.histogram(stds,bins=bins)
  hist = hist[0]
  plt.figure(0)
  plt.title('std (> better)')
  ranges = np.linspace(min_bin,max_bin,bins)
  plt.bar(ranges,hist)
  plt.show()

def capture():
   capture = cv2.VideoCapture(0)
   sucess,frame = capture.read()
   roi_width = int(frame.shape[0] * 0.5) #paisagem
   roi_height = int(frame.shape[1] * 0.5)
   middle_w = int(frame.shape[1]/2)
   middle_h = int(frame.shape[0]/2)
   x = middle_w  - int(roi_width / 2)
   y = middle_h - int(roi_height / 2)
   i = 1
   while (sucess):
     frame_cpy = frame.copy() 
     cv2.rectangle(frame_cpy,(x,y),(x+roi_width,y+roi_height),(0,255,0),1) #inverte em paisagem
     cv2.imshow('',frame_cpy)
     frame_cpy = None
     k = cv2.waitKey(30)
     if k == ord('q'):
        break
     gap = 10 
     if k == ord('l'): #salva foto de leite lata
        i = i + 1
        roi = frame[y:y+roi_height-gap,x:x+roi_width-gap]
        roi = cv2.resize(roi,(150,200),interpolation=cv2.INTER_AREA)
        cv2.imwrite('./datasets/captures/leite_lata/%d.jpg'%(i),roi)
     elif k == ord('c'): #salva foto de leite caixa
        i = i + 1
        roi = frame[y:y+roi_height-gap,x:x+roi_width-gap]
        roi = cv2.resize(roi,(150,200),interpolation=cv2.INTER_AREA)
        cv2.imwrite('./datasets/captures/leite_caixa/%d.jpg'%(i),roi)   
     elif k == ord('f'): 
        i = i + 1
        roi = frame[y:y+roi_height-gap,x:x+roi_width-gap]
        roi = cv2.resize(roi,(150,200),interpolation=cv2.INTER_AREA)
        cv2.imwrite('./datasets/captures/fermento/%d.jpg'%(i),roi)     
     sucess,frame = capture.read()     
   capture.release()   
   cv2.destroyAllWindows()

  


   
def chart_data(): 
    loader = FileLoader(r'C:\Users\Ivo Ribeiro\Documents\open-cv\datasets\captures')
    loader.load_files()
    seed = 11
    tsne = TSNE(n_components=2,random_state=seed)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors = ['b','g','r'] #3 classes
    all_features = []
    all_labels = []
    all_class_names = []

    for row in loader.files:
        imgs_fnames = row['imgs_per_class']
        features = []
        labels = []
        class_names = []
        for f_name in imgs_fnames:
            img = cv2.imread(f_name)
            descr = getDescriptor(img,expected_shape=(128,64)) 
            features.append(descr)
            labels.append(row['index'])
            class_names.append(row['class_name'])
        row['features'] = np.array(features)       
        all_features.extend(features)
        all_labels.extend(labels)
        all_class_names.extend(class_names)
    all_features = np.array(all_features)
    all_labels = np.array(all_labels)
  
    stds = np.std(all_features,axis=0)  
    all_features = all_features[:,stds > 0.0003]
    
    print(all_features.shape)
    print(all_labels.shape)

    for label,label_name in zip(np.unique(all_labels),np.unique(all_class_names)):
      features = all_features[ all_labels==label ,:]
      transformed = tsne.fit_transform(features)
      ax.scatter(x=transformed[:,0],y=transformed[:,1],c=colors[label],label=label_name)
    plt.title('captures dataset')
    ax.legend(loc='best')
    plt.show()


def main():
    print('1 para evaluate \n2 para real time test\n3 capturar\n4 show std\n5 chart')
    v = input()
    if v =='1':  
      evaluate_knn()
    elif v=='2':
      real_time_test()
    elif v=='3':
      capture()
    elif v=='4':
      show_std() 
    elif v=='5':
        chart_data()
    else:
      img = cv2.imread(r'C:\Users\Ivo Ribeiro\Documents\open-cv\datasets\captures\leite_lata\236.jpg') 
      img2 = cv2.imread(r'C:\Users\Ivo Ribeiro\Documents\open-cv\datasets\captures\leite_lata\290.jpg')    
      img = pre_process(img)
      img2= pre_process(img2)
      print(img.shape)
      while 1:
        k = cv2.waitKey(0)
        if k ==ord('q'):
          break 
        cv2.imshow('1',img)  
        cv2.imshow('2',img2)  
        

if __name__ == '__main__':
    main()
 
