import matplotlib.pyplot as plt
import os
import ast
import seaborn as sns
import pandas as pd
import numpy as np

def extract_exp_histogram(models_cam):
    
    path_corr = './histogram/'
    if not os.path.exists(path_corr):
        os.makedirs(path_corr)
        
  
    model = 'moto-droid-maxx'
        
    with open(f'./correlation/{model}_taken.txt', 'r') as f:
        taken = f.read().split(', ')
    taken = taken[0:-1]
    taken = [*map(lambda x: ast.literal_eval(x), taken)]
    taken = np.array(taken)
    taken = np.vstack((taken, np.ones(taken.shape[0]))).T
    taken = pd.DataFrame(taken, columns=['corr', 'capturada'])
   
    with open(f'./correlation/{model}_notTaken.txt', 'r') as f:
        not_taken = f.read().split(', ')
    not_taken = not_taken[0:-1]
    not_taken = [*map(lambda x: ast.literal_eval(x), not_taken)]
    not_taken = np.array(not_taken)
    
    np.random.shuffle(not_taken)
    not_taken = not_taken[np.random.choice(len(not_taken), size=taken.shape[0], replace=False)]
       
    not_taken = np.vstack((not_taken, np.zeros(not_taken.shape[0]))).T
    not_taken = pd.DataFrame(not_taken, columns=['corr', 'capturada'])


    testpd = taken.append(not_taken)
    print(len(testpd))
    # ['#1f77b4', '#ff7f0e',
    test1 = sns.distplot(x=taken['corr'], color='#1f77b4', bins=70)
    test1 = sns.distplot(x=not_taken['corr'], color='#ff7f0e', bins=70)

 
    #test1 = sns.histplot(data=testpd, x='corr', hue='capturada')  
   
    figure1 = test1.get_figure()    
  

    #figure.title(f'{model}')
    
    figure1.savefig(path_corr + f'{model}1.png', dpi=400)

    
    test = False
    figure = False
        
    
    
    