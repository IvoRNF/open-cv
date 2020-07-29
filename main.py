import numpy as np 
import cv2



class OpenCvTests:

  def __init__(self):
      pass
      
  
  

  
  def showContours(self): 
    img = np.zeros((200,200),dtype=np.uint8) 
    squareWidth = 100
    x,y = 50,50
    img[x:x+squareWidth,y:y+squareWidth] = 255
    img[10:25,10:25] = 255 
    _ , thresh = cv2.threshold(img,127,255,0)
    contours , hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contourColor = (0,255,0)
    contourSize = 3
    contourIdx = -1
    colorImg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    cv2.drawContours(colorImg,contours,contourIdx,contourColor,contourSize) 
    cv2.imshow('Contor the square ',colorImg)
    cv2.waitKey()
    cv2.destroyAllWindows()        

  def convertPhoto(self,source_file_name : str,dest_file_name : str):
    img = cv2.imread(source_file_name)
    cv2.imwrite(dest_file_name,img)
    
  def captureVideoCamera(self,seconds : int, file_name : str):
    camCapture = cv2.VideoCapture(0)
    fps = 30
    size = (int(camCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(camCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    videoWriter = cv2.VideoWriter(file_name,
                                   cv2.VideoWriter_fourcc('I','4','2','0'),
                                  fps,size)
    success,frame = camCapture.read()
    numFramesRemaining = seconds * fps - 1 
    while success and (numFramesRemaining>0):
        videoWriter.write(frame)
        success,frame = camCapture.read()
        numFramesRemaining -= 1
        
  def displayImageOnWindow(self, file_name : str):
     img = cv2.imread(file_name)
     print(type(img))
     print(img.dtype)
     cv2.imshow(file_name,img)
     cv2.waitKey()
     cv2.destroyAllWindows()
     
  def roundingCircles(self, file_name : str): 
     planets = cv2.imread(file_name)
     grayedImg = cv2.cvtColor(planets, cv2.COLOR_BGR2GRAY)
     grayedImg = cv2.medianBlur(grayedImg,5)
     circles = cv2.HoughCircles(grayedImg,cv2.HOUGH_GRADIENT, 1,120,param1=100,param2=30,minRadius=0,maxRadius=0)
     circles = np.uint16(np.around(circles))
     for elem in circles[0,:]: 
        cv2.circle(planets,(elem[0],elem[1]),elem[2],(0,255,0),2)
        cv2.circle(planets,(elem[0],elem[1]),2,(0,0,255),3)
     cv2.imshow('Hough circles',planets)
     cv2.waitKey()
     cv2.destroyAllWindows()     
  


  def removingBackgroundWathershed(self, file_name : str): 
    img = cv2.pyrDown(  cv2.imread(file_name) )
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret , thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    #remove the noise 
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations=2)
    #find the background
    sure_bg = cv2.dilate(opening,kernel,iterations=3)
    #find the foreground
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret , sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(),255,0)
    sure_fg = sure_fg.astype(np.uint8)
    #find the unknown region
    unknown = cv2.subtract(sure_bg,sure_fg)
    ret,markers = cv2.connectedComponents(sure_fg)
    markers += 1
    markers[unknown==255]=0
    markers = cv2.watershed(img,markers)
    img[markers==-1]=[255,0,0]
    cv2.imshow(file_name , img)   
    cv2.waitKey()
    cv2.destroyAllWindows()
    
  def removingBackgroundAndContour(self,file_name : str): #desennhado os contornos manualmente
    originalImg = cv2.pyrDown(cv2.imread(file_name))
    img = originalImg.copy()     
    x = 40
    y = 40
    w = 385
    h = 510
    rect = (x,y,x+w,y+h)
    mask = np.zeros(img.shape[:2],np.uint8)

    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
     
    mask2 = np.where((mask==cv2.GC_BGD)|(mask==cv2.GC_PR_BGD),0,1).astype(np.uint8)
    img = img * mask2[:,:,np.newaxis]
           
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

    contours , hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contourColor = (0,255,0)
    contourSize = 3
    contourIdx = 1

    cv2.drawContours(originalImg,contours,contourIdx,contourColor,contourSize) 
    
    #cv2.rectangle(originalImg,(x,y),(w,h),(0,255,0),3)
    cv2.imshow('',originalImg)
    cv2.waitKey()
    cv2.destroyAllWindows()
    
  def detectingFaces(self): 
    classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
    cam= cv2.VideoCapture(0)
    while (cv2.waitKey(1)==-1):
        success ,frame = cam.read()
        if success:    
           grayed = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
           faces = classifier.detectMultiScale(grayed,1.08,5,minSize=(100,100),maxSize=(300,300))    
           for (x,y,w,h) in faces: 
              frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
           cv2.imshow('',frame)    
    cv2.destroyAllWindows()    
def main():
   openCv = OpenCvTests()
   openCv.detectingFaces()
   #openCv.removingBackgroundAndContour('./livro.jpg') 
if __name__ == '__main__':
    main()

