import os
from PIL import Image
import numpy as np
from skimage.io import imsave


def extract_reference(models_cam):
    
    for model in models_cam:
        
        path_noises = './noise/' + model + '/'
        noises = os.listdir(path_noises)
        print(noises)
        w, h = Image.open(path_noises + noises[0]).size
        N = len(noises)
    
        arr = np.zeros((w, h), np.float)
    
        for im in noises:
            try:
                imarr = np.asarray(Image.open(path_noises + im))
                arr = arr + imarr
            except Exception as e:
                print(e)
        
        arr = arr / N
        
       
        path_reference = './reference/'
        if not os.path.exists(path_reference):
            os.makedirs(path_reference)
            
        imsave(arr=arr, fname= path_reference + model + '.jpg')