import cv2
import numpy as np
import os
import time
import math
import matplotlib.pyplot as plt 
from skimage.feature import hog
from skimage import exposure
from file_loader import FileLoader

class Knn(FileLoader):
    
    def __init__(self):
        super().__init__(dir_to_walk=r'C:\Users\Ivo Ribeiro\Documents\open-cv\datasets\captures')
        self.knn_fname = './my_knn.xml'
        self.loaded = False
        self.shape = (128,64)
        self.hog = self.createHog()
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
                descriptor = self.getHogDescriptor( cv2.imread(file_name,cv2.IMREAD_GRAYSCALE))
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
         #necessario aspect raio 1:2
        hog = cv2.HOGDescriptor()#(100,200),(8,8),(4,4),(8,8),9,1,-1,0,0.2,1,64,True)
        return hog            
    def train(self,data , labels):
        self.knn.train(data,cv2.ml.ROW_SAMPLE,labels)
        self.knn.save(self.knn_fname)
        print('saving knn to file')  
    
    def getHogDescriptor(self,sample,hog_open_cv=True):
      sampleToPredict = sample
      if len(sampleToPredict.shape)>2:
        sampleToPredict = remove_ilumination(sampleToPredict)
        sampleToPredict = cv2.cvtColor(sampleToPredict,cv2.COLOR_BGR2GRAY)  
      if sampleToPredict.shape != self.shape:
          reversedShape = self.shape[::-1]
          sampleToPredict = cv2.resize(sampleToPredict,reversedShape,interpolation=cv2.INTER_AREA)  
      if(hog_open_cv):
        descr = self.hog.compute(sampleToPredict) #opencv hog
        descr = np.squeeze(descr)
      else:
        descr =hog(sampleToPredict,orientations=8,pixels_per_cell=(4,4),
                            cells_per_block=(1,1),visualize=False,multichannel=False) #skimage hog      
      return descr
    def processAndPredict(self,sample , k = 6):
        descriptor = self.getHogDescriptor(sample)  
        return self.knn.findNearest(np.array([descriptor],dtype=np.float32), k)
    def pyrDown(self,img,levels=1):
        for i in range(levels):
            img = cv2.pyrDown(img)
        return img
def middleRects(shape,start=2,end=8,step=1,center_x=0,center_y=0):
   h,w = shape[:2]
   for i in np.arange(start,end,step):
      factor = i/10
      rect_width = int(h * factor) #inverte em paisagem
      rect_height = int(w * factor)
      x = (center_x - rect_width//2)
      y = (center_y - rect_height//2)     
      yield (x,y,rect_width,rect_height)
  
def real_time_test():
   min_distance = 170
   knn = Knn()
   knn.run()
   capture = cv2.VideoCapture(0)
   sucess,frame = capture.read()
   center_pt = (frame.shape[1]//2,frame.shape[0]//2)
   x_center,y_center = center_pt
   curr_rect = None
   sliding_rects = [ ]
   i = 0
   for curr_rect in middleRects(frame.shape,center_x=x_center,center_y=y_center):
      if i == 3:
        sliding_rects.append(list(curr_rect))
        break
      i+=1

   x,y,w,h = tuple(sliding_rects[0])
   x = w//4
   y = h//4
   x_step = 10
   y_step = 10
   for _ in np.arange(6):
      sliding_rects.append(list((x,y,w,h)))
      x += x_step
   x,y,w,h = curr_rect
   for _ in np.arange(6):
      y += y_step 
      for _ in np.arange(6):
        sliding_rects.append(list((x,y,w,h)))
        x += y_step
   sliding_rects = np.array(sliding_rects)
   sliding_rects_sz = sliding_rects.shape[0] 
   print(sliding_rects_sz)
   print(sliding_rects)
   i = 0
   found_rect_idx = -1
   while (sucess):
      frame_cpy = frame.copy() 
      cv2.circle(frame_cpy,(x_center,y_center),10,(0,255,255),1)
      
      k = cv2.waitKey(5)
      if k == ord('q'):
          break
      sucess,frame = capture.read()
      if(i==sliding_rects_sz):
        i = 0 
      x,y,w,h = tuple(sliding_rects[i])
      if(found_rect_idx == -1):
        center_rect = sliding_rects[0]
        _x,_y,_w,_h = center_rect
        cv2.rectangle(frame_cpy,(_x,_y),(_x+_w,_y+_h),(0,255,0),1)
        #i+=1
      roi = frame[y:y+h,x:x+w]
      #if(found_rect_idx == -1):
        #cv2.rectangle(frame_cpy,(x,y),(x+w,y+h),(255,0,0),1)  
      response = knn.processAndPredict(roi)
      distance = np.sum ( np.squeeze(response[3]) )
      
      if(distance < min_distance):
        found_rect_idx = i
        cv2.rectangle(frame_cpy,(x,y),(x+w,y+h),(0,255,0),1)  
        class_idx = int(response[0])
        class_name = knn.class_names[class_idx]
        txt = '%s(%.2f)' % (class_name,distance) 
        cv2.putText(frame_cpy,txt,(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2) 
      else:
        found_rect_idx = -1 
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
          print('%s %d classificado como %d - arquivo %s'
                % (class_name,class_idx,predicted_class_idx, os.path.basename(fname) ))
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
       descriptors.append(  knn.getHogDescriptor( cv2.imread(fname , cv2.IMREAD_GRAYSCALE) )  )
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
def sliding_window(frame, step=20, window_size=(100, 40)):
    img = frame 
    img_h, img_w = img.shape[:2]
    window_w, window_h = window_size
    for y in np.arange(0, img_w, step):
      for x in np.arange(0, img_h, step):
        yield (x, y, window_w,window_h)
def pyramid(img, scale_factor=1.25, min_size=(150,200),max_size=(600, 600)):
    h, w =  img.shape[:2]
    min_w, min_h = min_size
    max_w, max_h = max_size
    while w >= min_w and h >= min_h:
      if w <= max_w and h <= max_h:
        yield img
      w /= scale_factor
      h /= scale_factor
      img = cv2.resize(img, (int(w), int(h)),interpolation=cv2.INTER_LINEAR)  
def scale_rect(shape_origin, shape_dest,rect):
     x,y,w,h = rect 
     scaleH = shape_dest[0]/float(shape_origin[0])
     scaleX = shape_dest[1]/float(shape_origin[1])
     return  (int(x*scaleX),int(y*scaleH),int(w*scaleX),int(h*scaleH))
    
def detect_mult_scale(img,threashold=170,winSize=(70,100),winStep=20,knn=None): 
  h,w = img.shape[:2]
  result = None
  min_distance = 170
  win_w,win_h = winSize
  for resized in pyramid(img,1.25,(38,50),(w,h)):
     for (x,y,w,h) in sliding_window(resized,winStep,winSize):
         roi = resized[x:x+h,y:y+w]
         if (roi.shape[0]>=win_h) and(roi.shape[1]>=win_w):
           response = knn.processAndPredict(roi)
           distance = np.sum ( np.squeeze(response[3]) )
           if distance <  min_distance:
              predicted_class_idx = response[0]
              result = (predicted_class_idx,distance,scale_rect(resized.shape,img.shape,(x,y,w,h)))
              min_distance = distance
           if distance <= threashold:
              return result   
  return result         
def circular_center_points(frame): 
   middle_w = frame.shape[1]//2
   middle_h = frame.shape[0]//2
   for radius in np.arange(0,20,20):
    for degree in np.arange(0,360,80):
        x = middle_w + radius * math.cos(degree * math.pi/180)
        y = middle_h + radius * math.sin(degree * math.pi/180)
        yield (round(y),round(x))

def remove_ilumination(img): 
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV) 
    (h,s,v) = cv2.split(hsv) 
    s[:] = 0
    h[:] = 0
    hsvValueOnly = cv2.merge([h,s,v])
    converted = cv2.cvtColor(hsvValueOnly,cv2.COLOR_HSV2BGR) 
    return converted
def pyr(img , factor=0.15,levels=5): 
    h,w = img.shape[:2] 
    yield img.copy()
    for level in np.arange(levels):
      new_sz = (int(w-(w*factor)),int(h-(h*factor)))
      w,h = new_sz
      yield cv2.resize(img,new_sz)
def main():
    print('1 para evaluate \n2 para real time test\n3 capturar\n4 show std\n5 teste pyr')
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
      img = cv2.imread(r'C:\Users\Ivo Ribeiro\Documents\open-cv\datasets\testes_captures\2.jpg')
      
      rectAtLv2 = (130,60,150,220)
      while True: 
        i = 0
        for resized in pyr(img,levels=2): 
         if(i==2):
           x,y,w,h = rectAtLv2
           cv2.rectangle(resized,(x,y),(x+w,y+h),(0,255,0),2) 
         elif(i==0):
           origin_shape = 346, 462 #shape level 2 
           x,y,w,h = scale_rect(origin_shape,resized.shape[:2],rectAtLv2) #rect reescaled from level 2 to 1
           cv2.rectangle(resized,(x,y),(x+w,y+h),(0,255,0),2)    
         cv2.imshow(str(i),resized)
         i+= 1
        k = cv2.waitKey(10)
        if k==ord('q'):
          break 
      cv2.destroyAllWindows()  

if __name__ == '__main__':
    img = cv2.imread(r'C:\Users\Ivo Ribeiro\Documents\open-cv\datasets\testes_captures\2.jpg')
    main()
 
