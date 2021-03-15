#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 19:10:59 2021

@author: root
"""
from PIL import Image
import numpy as np
import pickle, os
from nproc import npc

models_cam = os.listdir('./dataset')

for model in models_cam:
    fit = pickle.load(open(f'./model/{model}', 'rb'))

    path_rpn = f'./reference/{model}.jpg'
    #path_spn = './noise/iphone-4s/(iP4s)2.jpg' ok
    #path_spn = './noise/htc-one-m7/(HTC-1-M7)4.jpg' ok
    
    
    
    rpn = Image.open(path_rpn)
    rpn = np.asarray(rpn)
    p = (rpn - rpn.mean()).flatten()
    
    spn = Image.open(path_spn)
    spn = np.asarray(spn)
    n = (spn - spn.mean()).flatten()
    
    corr = np.corrcoef(n, p)[0][1]
    
    corr = corr.reshape(-1, 1)
    
    test = npc()
    try:
        fitted_score = test.predict(fit, corr)
        print(model)
    except Exception as e:
        pass
    
    
    # ESSES NOISES TAMBÉM PARTICIPARAM PARA A CONSTRUÇÃO DO RPN