import numpy as np
from skimage.feature import hog
from skimage.feature import local_binary_pattern
import cv2



def remove_border_slices(img,fator=0.05):
    h,w = img.shape[:2]
    part_h = int(h * fator)
    part_w = int(w * fator)
    result = np.zeros(shape=(h-part_h*2,w-part_w*2))
    result = img[part_h:h-part_h,part_w:w-part_w]
    return cv2.resize(result,(w,h),interpolation=cv2.INTER_AREA)  

def remove_ilumination(img): 
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV) 
    (h,s,v) = cv2.split(hsv) 
    s[:] = 0
    h[:] = 0
    hsvValueOnly = cv2.merge([h,s,v])
    converted = cv2.cvtColor(hsvValueOnly,cv2.COLOR_HSV2BGR) 
    return converted

def getDescriptor(sample,expected_shape=(128,64) ,descr_open_cv=False,name='LBPH'):
      sampleToPredict = sample
      if len(sampleToPredict.shape)>2:
        sampleToPredict = remove_ilumination(sampleToPredict)
        sampleToPredict = cv2.cvtColor(sampleToPredict,cv2.COLOR_BGR2GRAY)  
      if sampleToPredict.shape != expected_shape:
          reversedShape = expected_shape[::-1]
          sampleToPredict = cv2.resize(sampleToPredict,reversedShape,interpolation=cv2.INTER_AREA)
      sampleToPredict = remove_border_slices(img=sampleToPredict,fator=0.07)      
      sampleToPredict = cv2.fastNlMeansDenoising(sampleToPredict)
      sampleToPredict = cv2.equalizeHist(sampleToPredict)
      descr = None
      if(descr_open_cv):
        if name=='ORB':
          descr_sz = 64
          descr = np.zeros(descr_sz) # tamanho max baseado em experimento
          orb = cv2.ORB_create(nfeatures=descr_sz)
          kp = orb.detect(sampleToPredict,None)
          kp,orb_desc = orb.compute(sampleToPredict,kp)
          if orb_desc is not None: 
              orb_desc = orb_desc.ravel()
              for i in range(orb_desc.shape[0]):
                 if i < descr_sz:
                    descr[i] = orb_desc[i]  
                 else:
                    break  
        elif name=='HOG_OPENCV':
          hog = cv2.HOGDescriptor()          
          descr = hog.compute(sampleToPredict) #opencv hog
          descr = np.squeeze(descr)
      else:
        if name=='HOG_SKIMAGE':
           descr =hog(sampleToPredict,orientations=8,pixels_per_cell=(4,4),
                            cells_per_block=(1,1),visualize=False,multichannel=False) #skimage hog 
        elif name=='LBPH':  
          descr = local_binary_pattern(image=sampleToPredict,P=8,R=1,method='default')
          descr = descr.ravel()
          hist,_ = np.histogram(descr,bins=np.arange(255),density=True)
          descr = hist                         
      return descr