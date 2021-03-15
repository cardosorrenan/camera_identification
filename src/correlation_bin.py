from PIL import Image
import numpy as np
import os


def extract_correlation_bin(models_cam):
    
    path_corr = './correlation_bin/'

    if not os.path.exists(path_corr):
        os.makedirs(path_corr)
        
    for model_i in models_cam:
        
        rpn = Image.open(f'./reference/{model_i}.jpg')
        rpn = np.asarray(rpn)
        p = (rpn - rpn.mean()).flatten()
        
        for model_j in models_cam:
            noises = os.listdir(f'./noise/{model_j}')
            corrs = []
                
            for noise in noises:
                
                spn = Image.open(f'./noise/{model_j}/{noise}')
                spn = np.asarray(spn)
                n = (spn - spn.mean()).flatten()
                corrs.append(np.corrcoef(n, p)[0][1])
             
            print(len(corrs))
            
            with open(path_corr + f'{model_i}_{model_j}.txt', 'w') as f:
                for item in corrs:
                    f.write("%s, " % item)
