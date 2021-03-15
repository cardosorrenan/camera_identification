from PIL import Image
import numpy as np
import os


def extract_correlation(models_cam):
    
    path_corr = './correlation/'

    if not os.path.exists(path_corr):
        os.makedirs(path_corr)
        
    for model_i in models_cam:
        
        taken_c = []
        not_taken_c = []
        path_rpn = f'./reference/{model_i}.jpg'
        
        
        for model_j in models_cam:
            noises = os.listdir(f'./noise/{model_j}')
                
            for noise in noises:
                
                path_spn = f'./reference/{model_i}.jpg'
                corr = calc_correlation(path_rpn, path_spn)
                     
                if model_i == model_j:
                    taken_c.append(corr)                     
                else:                
                    not_taken_c.append(corr)
                    
                    
        with open(path_corr + f'{model_i}_taken.txt', 'w') as f:
            for item in taken_c:
                f.write("%s, " % item)
            
        with open(path_corr + f'{model_i}_notTaken.txt', 'w') as f:
            for item in not_taken_c:
                f.write("%s, " % item)
    

def calc_correlation(path_rpn, path_spn):
    rpn = Image.open(path_rpn)
    rpn = np.asarray(rpn)
    p = (rpn - rpn.mean()).flatten()
    
    spn = Image.open(path_spn)
    spn = np.asarray(spn)
    n = (spn - spn.mean()).flatten()
    
    corr = np.corrcoef(n, p)[0][1]
    
    return corr
            
    
    
    
    
    
    
