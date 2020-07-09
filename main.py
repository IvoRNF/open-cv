import numpy as np 
import cv2



class OpenCvTests:

  def __init__(self):
      pass

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

       
def main():
   openCv = OpenCvTests()
   #openCv.displayImageOnWindow('livia.jpg')
   #openCv.captureVideoCamera(10,'output_video2.avi')

if __name__ == '__main__':
    main()

