import os
from PIL import Image
import numpy as np
from skimage.io import imsave


def extract_reference(model):
    path_noises = './noises/' + model + '/'
    noises = os.listdir(path_noises)

    w, h = Image.open(path_noises + noises[0]).size
    N = len(noises)

    arr = np.zeros((w, h, 3), np.float)

    for im in noises:
        try:
            imarr = np.asarray(Image.open(path_noises + im), dtype = np.float64)
            arr = arr + (imarr/N)
        except Exception as e:
            print(e)
    
    reference_arr = np.array(np.round(arr), dtype=np.float64)
    
    path_reference = './references/'
    if not os.path.exists(path_reference):
        os.makedirs(path_reference)
        
    imsave(arr=reference_arr, fname= path_reference + model + '.jpg')