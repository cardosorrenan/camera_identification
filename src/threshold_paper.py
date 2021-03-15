import ast
import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from nproc import npc
import os
from sklearn.model_selection import train_test_split


def pdf_gaussian(data):
    N = data.shape[0]
    u_x = data.mean()
    
    m1 = [*map(lambda x: abs(x - u_x), data)]
    m1 = np.array(m1).sum()/N
    
    m2 = [*map(lambda x: pow(x - u_x, 2), data)]
    m2 = np.array(m2).sum()/N
    
    param_beta = pow(m1, 2)/m2
    beta = 1/G(param_beta)
    
    alpha = m1 / (gamma(1/beta) * gamma(2/beta))
    
    pt1 = 1/(2*alpha*gamma(1/beta))
    pt22 = pow(abs(data - u_x)/alpha, beta)*-1 
    pt2 = np.exp(pt22)

    return pt1 * pt2


def G(x):
    y = pow(gamma(2/x), 2) / (gamma(1/x) * gamma(3/x))
    return y
    

def get_corrs_data(file):
    # a - RPN from model a
    # b - SPNs from model b
    with open(f'./correlation/{file}', 'r') as f:
        corr = f.read().split(', ')
        corr = corr[0:-1]
        corr = np.array([*map(lambda x: ast.literal_eval(x), corr)])
    return corr
    

def extract_threshold(models_cam):
    
    model_1 = models_cam[0]
    
    
    same = get_corrs_data(f'{model_1}_notTaken.txt')
    diff = get_corrs_data(f'{model_1}_taken.txt')
    
    same = np.sort(same)
    y_same = pdf_gaussian(same)
    diff = np.sort(diff)
    y_diff = pdf_gaussian(diff)
    
    plt.plot(same, y_same)
    plt.plot(diff, y_diff)
    