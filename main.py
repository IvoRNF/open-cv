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
     img = cv2.pyrDown( cv2.imread(file_name) )
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
    
  def removingBackgroundAndContour(self,file_name : str,rect): #desennhado os contornos manualmente
    originalImg = cv2.imread(file_name)
    img = originalImg.copy()     
    x,y,w,h = rect
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
    contourSize = 1
    contourIdx = 0

    cv2.drawContours(originalImg,contours,contourIdx,contourColor,contourSize) 
    
    #cv2.rectangle(originalImg,(x,y),(w,h),(0,255,0),1)
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
  def HarrisFeatureDetection(self,file_name : str):
    img = cv2.pyrDown( cv2.imread(file_name) )
    grayed = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    grayed = np.float32(grayed)
    dest = cv2.cornerHarris(grayed,5,3,0.04)
    img[dest>0.01*dest.max()]=[0,255,0]
    cv2.imshow(file_name,img)
    cv2.waitKey()
    cv2.destroyAllWindows()

  def trackObject(self):
    
    capture = cv2.VideoCapture(0)
    for i in range(10):
      captured,frame = capture.read()
    frame_h, frame_w = frame.shape[:2]

    w = frame_w//8
    h = frame_h//8
    x = frame_w//2 - w//2
    y = frame_h//2 - h//2
    track_window = (x,y,w,h)
    roi = frame[y:y+h,x:x+w]
    hsv_roi = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    mask = None
    roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
    cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
    term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
    captured,frame = capture.read()
    while(True):
      if not captured:
        break
      hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
      dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
      rotated_rect, track_window = cv2.CamShift(dst, track_window, term_crit)
      if not rotated_rect:
        break
      box_points = cv2.boxPoints(rotated_rect)
      box_points = np.int0(box_points)
      cv2.polylines(frame,[box_points],True,(0,255,0),2)
      cv2.imshow('cam shift',frame)
      k = cv2.waitKey(1)
      if k == 27:
        break
      captured,frame = capture.read()  
    cv2.destroyAllWindows()
    capture.release()

  def backgroundSubtractor(self):
    capture = cv2.VideoCapture(0)
    bkSubtractor = cv2.createBackgroundSubtractorKNN(detectShadows=False)
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,5))
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(17,11))
    captured,frame = capture.read()
    while(captured):
      fgmsk = bkSubtractor.apply(frame)
      _,thresh = cv2.threshold(fgmsk,244,255,cv2.THRESH_BINARY)
      cv2.erode(thresh, erode_kernel,thresh,iterations=2)
      cv2.dilate(thresh, dilate_kernel,thresh,iterations=2)
      contours, hierarchy = cv2.findContours(thresh , cv2.RETR_EXTERNAL,
                                             cv2.CHAIN_APPROX_SIMPLE)
      if(len(contours)>0):  
        biggest_contour = contours[0]
        for contour in contours:
            if cv2.contourArea(contour) > cv2.contourArea(biggest_contour):
                biggest_contour = contour
        x,y,w,h = cv2.boundingRect(biggest_contour)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)
      cv2.imshow('MOG BGR Subtractor',frame)
      k = cv2.waitKey(30)
      if k == 27:
        break
      captured,frame = capture.read()
    capture.release()
    cv2.destroyAllWindows()



  def mouse_move(self,event,x,y,flags,param):
      measure = np.array([[x],[y]],np.float32)
      if self.last_measure is None:
        self.kalman.statePre = np.array([[x],[y],[0],[0]],np.float32)
        self.kalman.statePost = np.array([[x],[y],[0],[0]],np.float32)
        prediction = measure
      else:
        self.kalman.correct(measure)  
        prediction = self.kalman.predict()
        cv2.line(self.img, (int(self.last_measure[0]), int(self.last_measure[1])),
         (int(measure[0]), int(measure[1])), (0, 255, 0))
        cv2.line(self.img, (int(self.last_prediction[0]), int(self.last_prediction[1])),
        (int(prediction[0]), int(prediction[1])), (0, 0, 255))   
      self.last_prediction = prediction.copy()
      self.last_measure = measure

  def trackingMouseWithKalman(self):
    self.img = np.zeros((800, 800, 3), np.uint8) 
    self.kalman = cv2.KalmanFilter(4,2)
    self.kalman.measurementMatrix = np.array(
                                        [[1, 0, 0, 0],
                                        [0, 1, 0, 0]], np.float32)
    self.kalman.transitionMatrix = np.array(
                                        [[1, 0, 1, 0],
                                        [0, 1, 0, 1],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]], np.float32)
    self.kalman.processNoiseCov = np.array(
                                        [[1, 0, 0, 0],
                                        [0, 1, 0, 0],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]], np.float32) * 0.03 

    self.last_measure = None 
    self.last_prediction = None 
    name = 'kalman_tracker'
    cv2.namedWindow(name)
    cv2.setMouseCallback(name,self.mouse_move)
    while(True):
      cv2.imshow(name,self.img)
      k = cv2.waitKey(1)
      if k == 27:
        break
    cv2.destroyAllWindows()
def main():
   openCv = OpenCvTests()
   openCv.backgroundSubtractor()
   #openCv.trackingMouseWithKalman()
   #openCv.trackObject()
   #openCv.HarrisFeatureDetection('./livro.jpg')
   #openCv.detectingFaces()
   #openCv.removingBackgroundAndContour('./banana100X100.jpg',(10,25,80,80)) 
   
if __name__ == '__main__':
    main()

