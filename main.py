# -*- coding: utf-8 -*-

import urllib.request
import pywt
import os
from skimage.io import imsave
from skimage.util import crop
import numpy as np
from PIL import Image

from src.filtering import apply_filter


def main():
        
    path_dataset = './dataset'
    models_cam = os.listdir(path_dataset)
    
    # only 1 model
    models_cam = models_cam[0:1]
    
    
    for model in models_cam:
        
        path_model = path_dataset + '/' + model
        images = os.listdir(path_model)
        
        # Only 10
        images = images[0:10]
        
        for image_idx, image in enumerate(images):

            path_cam = path_model + '/' + image
 
            im = Image.open(path_cam)
            img_2array = np.asarray(im)
            img_crop = img_2array[1500:2012, 1000:1512]
        
            result_final = np.zeros(img_crop.shape)
        
            for ch in range(img_crop.shape[2]): # Iterate over 3 color channels
                result_channel = []
                wavelet = pywt.wavedec2(img_crop[:, :, ch], 'db8', level=4)
                wavelet_details = wavelet[1:]
        
                for i, w_details_lvl in enumerate(wavelet_details): # Iterate over 4 details levels (cD4, cD3, cD2, cD1)
                    result_lvl = []
              
                    for j, coeff in enumerate(w_details_lvl): # Iterate over 3 sub-bands (hor, ver, dia)
                        
                        filtered_coeff = apply_filter(coeff)
                        result_lvl.append(filtered_coeff)
        
                    result_channel.append(tuple(result_lvl))
        
                result_channel.insert(0,  wavelet[0])
                wrec = pywt.waverec2(result_channel, 'db8')
                result_final[:, :, ch] = wrec
                
            noise = img_crop - result_final
            
            path_noise = './noises/' + model + '/'
            
            if not os.path.exists(path_noise):
                os.makedirs(path_noise)
            imsave(arr=noise, fname=path_noise + image)
            
            
if __name__ == '__main__':
    main()