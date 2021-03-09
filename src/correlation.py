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
                    taken_c.append(corr)                     
                else:                
                    not_taken_c.append(corr)
                    
                    
        with open(path_corr + f'{model_i}_taken.txt', 'w') as f:
            for item in taken_c:
                f.write("%s, " % item)
            
        with open(path_corr + f'{model_i}_notTaken.txt', 'w') as f:
            for item in not_taken_c:
                f.write("%s, " % item)

    
        
    
    
    
    
    
    
