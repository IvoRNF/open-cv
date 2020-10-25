import os 
import numpy as np


class FileLoader: 

    def __init__(self,dir_to_walk:str):
        self.files = []
        self.files_test = []
        self.PERC_TO_TEST = 0.2  #20 por cento para teste  
        self._dir_to_walk = dir_to_walk
        self.class_names = os.listdir(self._dir_to_walk)
    def load_files(self):          
      i = 0
      for root,dirs,files in os.walk(self._dir_to_walk):
          for name in files:
            basename = os.path.basename(root)
            y = self.files_contains_class_name_index_of(basename)
            if (y == -1):
              y = len(self.files)   
              self.files.append({"index":i,"imgs_per_class":[],"class_name":basename})
              i += 1 
            row = self.files[y]
            files_per_class = row["imgs_per_class"]
            files_per_class.append(os.path.join(root,name))
      for row in self.files:
          imgs = row['imgs_per_class']
          np.random.shuffle(imgs)
          len_test = int(len(imgs)*self.PERC_TO_TEST)
          imgs_test ,row['imgs_per_class'] = np.split(imgs,[len_test])
          self.files_test.append({"index":row['index'],"imgs_per_class":imgs_test,"class_name":row['class_name']})    
    def files_contains_class_name_index_of(self, name : str):
       i = 0
       files = self.files
       for f in files:
           row = files[i]
           if name == row['class_name']:
               return i
           i += 1    
       return -1       