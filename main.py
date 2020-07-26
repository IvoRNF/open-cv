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
     
  def removingBackground(self,file_name : str): #desennhado os contornos manualmente
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
       
    #newMask = cv2.pyrDown(cv2.imread('./banana-mask.jpg'))
    #newMask = cv2.cvtColor(newMask,cv2.COLOR_BGR2GRAY)
    #mask2[newMask==0] = 0
    
    #mask2, bgdModel, fgdModel = cv2.grabCut(img,mask,None,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)
    #img = img * mask2[:,:,np.newaxis]
    
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
       
def main():
   openCv = OpenCvTests()
   openCv.removingBackground('./livro.jpg')

if __name__ == '__main__':
    main()

