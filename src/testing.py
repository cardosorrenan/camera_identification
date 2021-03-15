import numpy as np
import pandas as pd
import ast
import os
from .noise import extract_noise_single_image
from .correlation import calc_correlation
import matplotlib.pyplot as plt
from scipy.stats import gennorm
from sklearn.cluster import KMeans

def testing_new(models_cam):
    
    #path_spn = './noise/sony-nex-7/(Nex7)265.JPG'
    path_spn = './noise/samsung-note-3/(GalaxyN3)274.jpg'
    #path_spn = './noise/htc-one-m7/(HTC-1-M7)274.jpg'
    result = []
    
    for model_i in models_cam:
        victories = 0
        path_rpn = f'./reference/{model_i}.jpg'
        noise = calc_correlation(path_spn, path_rpn)
        noise = np.array(noise).reshape(-1, 1)
        
        for model_j in models_cam:
            
            if (model_i != model_j or model_j != model_j):
                with open(f'./correlation_bin/{model_i}_{model_i}.txt', 'r') as f:
                    X = f.read().split(', ')
                    X = X[0:-1]
                    X = np.array([*map(lambda x: ast.literal_eval(x), X)])
                        
                with open(f'./correlation_bin/{model_i}_{model_j}.txt', 'r') as f:
                    Y = f.read().split(', ')
                    Y = Y[0:-1]
                    Y = np.array([*map(lambda x: ast.literal_eval(x), Y)])
                
                values = np.append(X, Y)
                plt.plot(X, Y)
                
                return
                kmeans = KMeans(n_clusters=2, random_state=0).fit(values.reshape(-1, 1))
                print(f'- correlacao imagem x rpn {model_i}: {round(noise[0][0], 4)}')
                print(f'- center 0 {model_i}: {round(kmeans.cluster_centers_[0][0], 4)}')
                print(f'- center 1 {model_j}: {round(kmeans.cluster_centers_[1][0], 4)}')
                res = kmeans.predict(noise)
                if (res == [0]):
                    victories += 1
                    print(f'- venceu: {model_i}')
                else:
                    print(f'- venceu: {model_j}')
                print()
                
        result.append({ 'model': model_i, 'victories': victories, 'noise': noise[0][0] })
    
    pd.DataFrame(result)
    
    
        
    print(pd.DataFrame(result))

            


