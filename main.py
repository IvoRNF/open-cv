import numpy as np 
import cv2



class OpenCvTests:

  def __init__(self):
      pass
      
  
  

  
  def showContours(self,img : np.ndarray): 
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
     
  def showContours2(self,file_name : str): #desennhado os contornos manualmente
    img = cv2.pyrDown(cv2.imread(file_name,cv2.IMREAD_UNCHANGED))
    
    #grayedImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret , thresh = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    
    contours , hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    colored = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    #cv2.drawContours(colored,contours,1,(0,255,0),3)
    
    #for contour in contours: 
    contour = contours[1]
    #print('contour at idx 1 ',contour)
    x,y,w,h = cv2.boundingRect(contour)
    #print('convert to a rect', x,y,w,h)     
    cv2.rectangle(colored,(x,y),(x+w,y+h),(0,255,0),2)
    (x,y),radius = cv2.minEnclosingCircle(contour)
    
    center = (int(x),int(y))
    radius = int(radius)
    cv2.circle(colored,center,radius,(0,0,255),2)
    
    cv2.imshow('',colored)
    cv2.waitKey()
    cv2.destroyAllWindows()
       
def main():
   openCv = OpenCvTests()
   # img = np.zeros((200,200),dtype=np.uint8) 
   # squareWidth = 100
   # x,y = 50,50
   # img[x:x+squareWidth,y:y+squareWidth] = 255
   # img[10:25,10:25] = 255 
   # openCv.showContours(img)
   openCv.showContours2('./yy.jpg')

if __name__ == '__main__':
    main()

