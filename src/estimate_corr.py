from PIL import Image
import numpy as np
from numpy import linalg as LA
import os
import json


def estimate_corr(models_cam):
    
    opts = dict.fromkeys(('taken', 'not_taken'), [])    
    corr_cam = dict.fromkeys(tuple(models_cam), opts)
    
    for model_i in models_cam:
               
        rpn = Image.open(f'./references/{model_i}.jpg')
        rpn = np.asarray(rpn)
        p = (rpn - rpn.mean()).flatten()
        
        for model_j in models_cam:
            noises = os.listdir(f'./noises/{model_j}')
                
            for noise in noises:
                 print(noise)
                 spn = Image.open(f'./noises/{model_j}/{noise}')
                 spn = np.asarray(spn)
                 n = (spn - spn.mean()).flatten()
                 corr = np.corrcoef(n, p)[0][1]
                 
                 
            if model_i == model_j:
                corr_cam[model_i]['taken'].append(corr)                     
            else:                
                corr_cam[model_i]['not_taken'].append(corr)

                             
                
    
    with open('result.json', 'w') as fp:
        json.dump(corr_cam, fp)

            
        
        
    
    
    
    
    
    
