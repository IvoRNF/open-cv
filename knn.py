import cv2
import numpy as np
import os
import time
import math
import matplotlib.pyplot as plt 
from skimage.feature import hog
from skimage import exposure

class Knn:
    
    def __init__(self):
        self.files = []
        self.files_test = []
        self.PERC_TO_TEST = 0.2  #20 por cento para teste  
        self.dir_to_walk = r'C:\Users\Ivo Ribeiro\Documents\open-cv\datasets\captures'
        self.knn_fname = './my_knn.xml'
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
                descriptor = self.getHogDescriptor( cv2.imread(file_name,cv2.IMREAD_GRAYSCALE) ,0)
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
        ''' 
        w,h = self.shape[::-1] #necessario aspect raio 1:2 , ajustando .. 
        metade = h//2 
        diff = w - metade 
        w = w - diff
        print('hog=')
        print(w,h)
        '''
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
        print('saving knn to file')  
    
    def getHogDescriptor(self,sample,pyrDownLevels=0):
      sampleToPredict = sample
      if len(sampleToPredict.shape)>2:
        sampleToPredict = remove_ilumination(sampleToPredict)
        sampleToPredict = cv2.cvtColor(sampleToPredict,cv2.COLOR_BGR2GRAY)
      if sampleToPredict.shape != self.shape:
          reversedShape = self.shape[::-1]
          sampleToPredict = cv2.resize(sampleToPredict,reversedShape,interpolation=cv2.INTER_AREA)     
      sampleToPredict = self.pyrDown(sampleToPredict,pyrDownLevels)
      descr = hog(sampleToPredict,orientations=8,pixels_per_cell=(16,16),
                            cells_per_block=(1,1),visualize=False,multichannel=False) #skimage hog
      #self.hog.compute(sampleToPredict) #opencv hog
      descr = np.squeeze(descr) 
      return descr
    def processAndPredict(self,sample , k = 12,pyrDownLevels=0):
        descriptor = self.getHogDescriptor(sample,pyrDownLevels)  
        return self.knn.findNearest(np.array([descriptor],dtype=np.float32), 12)
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
   min_distance = 26.00
   knn = Knn()
   knn.run()
   capture = cv2.VideoCapture(0)
   sucess,frame = capture.read()
   center_pt = (frame.shape[1]//2,frame.shape[0]//2)
   x_center,y_center = center_pt
   while (sucess):
      frame_cpy = frame.copy() 
      cv2.circle(frame_cpy,(x_center,y_center),10,(0,255,255),1)
      cv2.imshow('',frame_cpy)
      k = cv2.waitKey(5)
      if k == ord('q'):
          break
      sucess,frame = capture.read()
      #for (center_x,center_y) in circular_center_points(frame):
      for (x,y,w,h) in middleRects(frame.shape,center_x=x_center,center_y=y_center):
            roi = frame[y:y+h,x:x+w] 
            response = knn.processAndPredict(roi)
            distance = np.sum ( np.squeeze(response[3]) )
            class_idx = int(response[0])
            if(distance < min_distance):
              class_name = knn.class_names[class_idx]
              txt = '%s(%.2f)' % (class_name,distance)
              cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
              cv2.putText(frame,txt,(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2) 
              break
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
      img = cv2.resize(img, (int(w), int(h)),interpolation=cv2.INTER_AREA)  
def scale_rect(shape_origin, shape_dest,rect):
     x,y,w,h = rect 
     scaleH = shape_dest[0]/float(shape_origin[0])
     scaleX = shape_dest[1]/float(shape_origin[1])
     return  (int(x*scaleX),int(y*scaleH),int(w*scaleX),int(h*scaleH))

def detect_mult_scale(img,threashold=20000.00,winSize=(70,100),winStep=20,knn=None): 
  h,w = img.shape[:2]
  result = None
  min_distance = 40000.00
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
def teste_descriptor(): 
    img1 = cv2.imread(r'C:\Users\Ivo Ribeiro\Documents\open-cv\datasets\captures\leite_lata\32.jpg',cv2.IMREAD_GRAYSCALE)
    #img2 = cv2.imread(r'C:\Users\Ivo Ribeiro\Documents\open-cv\datasets\captures\leite_lata\340.jpg',cv2.IMREAD_UNCHANGED)
    print(img1.shape)
    #knn = Knn()
    #descr = knn.getHogDescriptor(img1)
    #print(descr.shape)
   
    
    descr = hog(img1,orientations=8,pixels_per_cell=(12,12),
                            cells_per_block=(1,1),visualize=False,multichannel=False)
    #hog_img_scaled = exposure.rescale_intensity(hog_img,in_range=(0,10))           


    print(descr.shape)         
    #while(True):
      #cv2.imshow('',hog_img_scaled)
      #cv2.waitKey(10)

def main():
    print('1 para evaluate \n2 para real time test\n3 capturar\n4 show std\n5 teste descriptor')
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
      teste_descriptor()   


if __name__ == '__main__':
    main()
 
