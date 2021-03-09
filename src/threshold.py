import matplotlib.pyplot as plt
import os
import ast


def extract_threshold(models_cam):
    
    path_corr = './histogram/'
    if not os.path.exists(path_corr):
        os.makedirs(path_corr)
        
    opts = ['taken', 'notTaken']    
    
    for model in models_cam:
        for opt in opts:
            with open(f'./correlation/{model}_{opt}.txt', 'r') as f:
                file = f.read().split(', ')
            file = file[0:-1]
            file = map(lambda x: ast.literal_eval(x), file)
            file = list(file) 
            color = "blue" if (opt == 'taken') else "red"
            plt.hist(file, bins='auto', color=f'{color}')
            
        plt.title(f'{model}')
        plt.savefig(path_corr + f'{model}.jpg')
        
    
    
    