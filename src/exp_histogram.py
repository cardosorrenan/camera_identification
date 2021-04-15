import os
import ast
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def make_plot(data, model, color):
    fig1 = sns.distplot(x=data['corr'], color=color, bins=50)
    fig1.set(xlabel=model)
    fig1.set(ylabel=None)

    
def extract_exp_histogram(models_cam):
    path_corr = './histogram/'
    if not os.path.exists(path_corr):
        os.makedirs(path_corr)
            
    for model in models_cam:
        plt.figure()
        
        with open(f'./correlation/{model}_taken.txt', 'r') as f:
            taken = f.read().split('\n')
        
        taken = [*map(lambda x: x.split(', ')[0], taken)]
        taken = taken[0:-1]
        taken = np.array([*map(lambda x: ast.literal_eval(x), taken)])
        taken = np.array(taken)
        taken = np.vstack((taken, np.ones(taken.shape[0]))).T
        taken = pd.DataFrame(taken, columns=['corr', 'capturada'])
        make_plot(taken, model, '#1f77b4')


        with open(f'./correlation/{model}_not_taken.txt', 'r') as f:
            not_taken = f.read().split('\n')
            
        not_taken = [*map(lambda x: x.split(', ')[0], not_taken)]
        not_taken = not_taken[0:-1]
        not_taken = np.array([*map(lambda x: ast.literal_eval(x), not_taken)])
        not_taken = np.array(not_taken)
        not_taken = np.vstack((not_taken, np.ones(not_taken.shape[0]))).T
        not_taken = pd.DataFrame(not_taken, columns=['corr', 'capturada'])
        make_plot(not_taken, model, '#ff7f0e')
        
        plt.savefig(f'histogram/{model}.jpg')
        
       
        
    
    
    