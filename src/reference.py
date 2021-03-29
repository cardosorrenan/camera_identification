import os, shutil
from PIL import Image
import numpy as np
from skimage.io import imsave


def extract_reference(models_cam):
    
    shutil.rmtree('./reference/', ignore_errors=True)
    
    if not os.path.exists('./reference/'):
        os.makedirs('./reference/')
    
    for model in models_cam:
        
        noises = os.listdir(f'./noise/train/{model}/')
        
        w, h = Image.open(f'./noise/train/{model}/{noises[0]}').size
        arr = np.zeros((w, h), np.float)
        num_noises = 0
    
        for noise in noises:
            try:
                imarr = np.asarray(Image.open(f'./noise/train/{model}/{noise}'))
                arr = arr + imarr
                num_noises += 1
            except Exception as e:
                print(e)
                continue
            
        
        arr = arr / num_noises
        imsave(arr=arr, fname=f'./reference/{model}.jpg')