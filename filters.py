import numpy as np
import cv2

class VConvolutionFilter:

    def __init__(self,kernel : np.ndarray):
       self._kernel = kernel
    def apply(self,src : np.ndarray,dest : np.ndarray):
       #-1 = use the name per channel depth of the origin img
       cv2.filter2D(src,-1,self._kernel, dest)

class SharpenFilter(VConvolutionFilter):
    def __init__(self):
        kernel = np.array([
             [-1,-1,-1],
             [-1,9,-1],
             [-1,-1,-1]
            ])
        VConvolutionFilter.__init__(self,kernel)

class BlurFilter(VConvolutionFilter):
    def __init__(self): 
       kernel = np.array([
                         [0.04,0.04,0.04,0.04,0.04],
                         [0.04,0.04,0.04,0.04,0.04],
                         [0.04,0.04,0.04,0.04,0.04],
                         [0.04,0.04,0.04,0.04,0.04],
                         [0.04,0.04,0.04,0.04,0.04]
                         ])
       VConvolutionFilter.__init__(self,kernel)

class EmbossFilter(VConvolutionFilter):
    def __init__(self):
       kernel = np.array([
       [-2,-1,0],
       [-1,1,1],
       [0,1,2]
       ])  
       VConvolutionFilter.__init__(self,kernel)

def strokeEdges(src : np.ndarray,dest : np.ndarray,blurKSize = 7,edgeKSize = 5):
    if blurKSize >= 3:
        blurredSrc =  cv2.medianBlur(src,blurKSize)
        graySrc = cv2.cvtColor(blurredSrc, cv2.COLOR_BGR2GRAY)
    else:
        graySrc = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
    cv2.Laplacian(graySrc, cv2.CV_8U,graySrc, ksize = edgeKSize)  
    normalizedInverseAlpha = (1.0/255)*(255 - graySrc)
    channels = cv2.split(src)
    for channel in channels:
       channel[:] = channel * normalizedInverseAlpha
    cv2.merge(channels,dest)

if __name__ == '__main__':     
   img = cv2.imread('./livia.jpg')
   strokeEdges(img,img)
   cv2.imshow('strokeEdges filter teste',img)
   cv2.waitKey()
   cv2.destroyAllWindows()
    
