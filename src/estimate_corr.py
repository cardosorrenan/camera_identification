from PIL import Image
import numpy as np
from numpy import linalg as LA

def estimate_corr(models_cam):
    
    models_cam = models_cam[0:1]
    
    for model in models_cam:
        
        # 1. P = GET THE CORR FOR THE ALL NOISES (CAMERA 'C')
        # 2. P' = GET THE CORR FOR THE ALL NOISES (NOT CAMERA 'C')  
        # 3. TRANSFORM P e P' IN DISTRIBUITON 
        # 4. GET THRESHOLD NEYMAN PEARSON
        
        
        path_noise = f'./noises/{model}/(iP4s)248.jpg'
        noise = Image.open(path_noise)
        noise = np.asarray(noise)
        
        path_reference = f'./references/{model}.jpg'
        reference = Image.open(path_reference)
        reference = np.asarray(noise)
        
        n = noise
        n_mean = noise.mean()
        aux_n = n - n_mean
        
        p = reference
        p_mean = reference.mean()
        aux_p = p - p_mean
    
        corr = np.dot(aux_n, aux_p.T) / ( LA.norm(aux_n) * LA.norm(aux_p) )
        
        
        print(corr.shape)
    
    
    
    
    
    
