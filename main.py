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
     
  def showContours2(self,file_name : str): #desennhado os contornos manualmente
    img = cv2.pyrDown(cv2.imread(file_name,cv2.IMREAD_UNCHANGED))
    
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #img = cv2.Canny(img,10,20)
    ret , thresholded = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    
    contours , hierarchy = cv2.findContours(thresholded,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img,contours,-1,(0,255,0),1)
    
     
    # contour = contours[1]   
    # x,y,w,h = cv2.boundingRect(contour)    
    # cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    # (x,y),radius = cv2.minEnclosingCircle(contour)
    
    # center = (int(x),int(y))
    # radius = int(radius)
    # cv2.circle(img,center,radius,(0,0,255),2)
    
    cv2.imshow('',img)
    cv2.waitKey()
    cv2.destroyAllWindows()
       
def main():
   openCv = OpenCvTests()
   openCv.roundingCircles('./planets.jpg')

if __name__ == '__main__':
    main()

