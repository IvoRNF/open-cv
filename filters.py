import numpy as np
import cv2 

def strokeEdges(src : np.array,dest : np.array,blurKSize = 7,edgeKSize = 5):
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
   dest = np.ones(shape=img.shape)
   strokeEdges(img,dest)
   cv2.imshow('stroking edges',img)
   cv2.waitKey()
   cv2.destroyAllWindows()
    
