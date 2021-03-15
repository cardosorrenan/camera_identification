import ast
import numpy as np
from scipy.special import gamma
from scipy.stats import gennorm
import matplotlib.pyplot as plt


def get_beta(x):
    N = x.shape[0]
    u_x = x.mean()
    m1 = [*map(lambda x: abs(x - u_x), x)]
    m1 = np.array(m1).sum()/N    
    m2 = [*map(lambda x: pow(x - u_x, 2), x)]
    m2 = np.array(m2).sum()/N    
    param_beta = pow(m1, 2)/m2
    beta = 1/G(param_beta)
    alpha = m1 / (gamma(1/beta) * gamma(2/beta))
    return beta 


def G(x):
    y = pow(gamma(2/x), 2) / (gamma(1/x) * gamma(3/x))
    return y
    

def get_corrs_data(a, b):
    # a - RPN from model a
    # b - SPNs from model b
    with open(f'./correlation_bin/{a}_{b}.txt', 'r') as f:
        corr = f.read().split(', ')
        corr = corr[0:-1]
        corr = np.array([*map(lambda x: ast.literal_eval(x), corr)])
    return corr
    

def extract_threshold(models_cam):
    
    model_1 = models_cam[0]
    model_2 = models_cam[1]
    
    same = get_corrs_data(model_1, model_1)
    diff = get_corrs_data(model_1, model_2)
    
    beta_same = get_beta(same)
    beta_diff = get_beta(diff)  

    x_same = np.linspace(gennorm.ppf(0.01, beta_same),
                gennorm.ppf(0.99, beta_same), 274)
    x_diff = np.linspace(gennorm.ppf(0.01, beta_diff),
                gennorm.ppf(0.99, beta_diff), 274)

    plt.plot(x_same, gennorm.pdf(x_same, beta_same))
    plt.plot(x_diff, gennorm.pdf(x_diff, beta_diff))
    
    # https://stackoverflow.com/questions/10138085/python-plot-normal-distribution
    
    
    
    
    