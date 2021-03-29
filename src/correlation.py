from PIL import Image
import numpy as np
import os, shutil


def extract_correlation(models_cam):
    
    shutil.rmtree('./correlation/', ignore_errors=True)
    
    if not os.path.exists('./correlation/'):
        os.makedirs('./correlation/')
    
    for model_i in models_cam:
        
        taken_c = []
        not_taken_c = []
        path_rpn = f'./reference/{model_i}.jpg'
        
        for model_j in models_cam:
            
            if model_i == model_j:
                noises = os.listdir(f'./noise/test/{model_j}')
                for noise in noises:
                    path_spn = f'./noise/test/{model_j}/{noise}'
                    corr = calc_correlation(path_rpn, path_spn) 
                    taken_c.append(f'{corr}, {noise}') 
                
            else:         
                noises = os.listdir(f'./noise/train/{model_j}')
                for noise in noises:
                    path_spn = f'./noise/train/{model_j}/{noise}'
                    corr = calc_correlation(path_rpn, path_spn)
                    not_taken_c.append(f'{corr}, {noise}')
                    
                noises = os.listdir(f'./noise/test/{model_j}')
                for noise in noises:
                    path_spn = f'./noise/test/{model_j}/{noise}'
                    corr = calc_correlation(path_rpn, path_spn)
                    not_taken_c.append(f'{corr}, {noise}')    
            
            print(f'{model_i} - {model_j} OK')
        
        with open(f'./correlation/{model_i}_taken.txt', 'w') as f:
            for item in taken_c:
                f.write("%s\n" % item)
            
        with open(f'./correlation/{model_i}_not_taken.txt', 'w') as f:
            for item in not_taken_c:
                f.write("%s\n" % item)
        


def calc_correlation(path_rpn, path_spn):
    rpn = Image.open(path_rpn)
    rpn = np.asarray(rpn)
    p = (rpn - rpn.mean()).flatten()
    
    spn = Image.open(path_spn)
    spn = np.asarray(spn)
    n = (spn - spn.mean()).flatten()
    
    corr = np.corrcoef(n, p)[0][1]
    
    return corr
            
    
    
    
    
    
    
