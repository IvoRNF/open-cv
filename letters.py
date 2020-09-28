def createLetterImg(letter : str,shape=(150,200),fontScale=5):
      h,w = shape
      img = np.zeros((h,w))
      middle_w = int(w/2)
      x = middle_w - int(w/4)
      middle_h = int(h/2)
      y = middle_h + int(h/4)
      txt = letter
      cv2.putText(img,txt,(x,y),cv2.FONT_HERSHEY_SIMPLEX,fontScale,(255,255,255),2)
      return img

def createLettersDataset(dir_to_save:str,samples_per_class=3,shape=(150,200),fontScale=5,end_letter='Z'):
   for i in range(ord('A'),ord(end_letter)+1):
      letter = chr(i)
      img = createLetterImg(letter,shape,fontScale)
      for j in range(samples_per_class):
         fname = os.path.join(dir_to_save,letter)
         directory = fname
         if not os.path.exists(directory):
             os.makedirs(directory)
         fname = os.path.join(directory,'%d.jpg' %(j))
         print('writing.. %s\n' % (fname))
         cv2.imwrite(fname,img)
   print('fim') 