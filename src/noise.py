import pywt, os, shutil
import numpy as np
from PIL import Image
from skimage.io import imsave
import random


def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def get_sigma(coeff, W_values, sigma_0):
    values = []
    for w in W_values:
        x = coeff**2 - sigma_0**2
        x_sum = x.sum()/w**2
        values.append(x_sum if x_sum > 0 else 0)        
    sigma = min(values)        
    return sigma


def wiener_filter(coeff, sigma, sigma_0):
    filtered_coeff = coeff*(sigma/(sigma + sigma_0**2))
    return filtered_coeff
                       
                        
def apply_filter(coeff):
    sigma_0 = 5
    W = [3, 5, 7, 9]
    sigma = get_sigma(coeff, W, sigma_0)
    coeff_filtered = wiener_filter(coeff, sigma, sigma_0)
    return coeff_filtered



def extract_noise(models_cam):
    
    shutil.rmtree('./noise/', ignore_errors=True)
    
    for model in models_cam: # Iterate over camera models
            
        images = os.listdir(f'./dataset/{model}/')
        
        random.shuffle(images)
        train_images = images[0:75]
        test_images = images[75:]
        
        for image in train_images: # Iterate over camera images
        
            if not os.path.exists(f'./noise/train/{model}/'):
                os.makedirs(f'./noise/train/{model}/')
                
            path_image =  f'{model}/{image}'
            extract_spn(path_image, 'train')
            
        for image in test_images: # Iterate over camera images
        
            if not os.path.exists(f'./noise/test/{model}/'):
                os.makedirs(f'./noise/test/{model}/')
                
            path_image =  f'{model}/{image}'
            extract_spn(path_image, 'test')
            

def extract_spn(path, mode):
    img_orig = np.asarray(Image.open('./dataset/' + path))
    img_crop = img_orig[1500:2012, 1000:1512]        
    img = img_crop
    
    img_denoised = np.zeros((img.shape[0], img.shape[1], 3))
    
    try:
        for ch in range(img.shape[2]): # Iterate over 3 color channels
            
            result_channel = []
        
            wavelet = pywt.wavedec2(img[:, :, ch], 'db8', level=4)
            wavelet_details = wavelet[1:]
        
            for i, w_details_lvl in enumerate(wavelet_details): # Iterate over 4 details levels (cD4, cD3, cD2, cD1)
                result_lvl = []
          
                for j, coeff in enumerate(w_details_lvl): # Iterate over 3 sub-bands (hor, ver, dia)
                    filtered_coeff = apply_filter(coeff)
                    result_lvl.append(filtered_coeff)
        
                result_channel.append(tuple(result_lvl))
    
            result_channel.insert(0,  wavelet[0])
            wrec = pywt.waverec2(result_channel, 'db8')
            img_denoised[:, :, ch] = wrec
            
        noise_arr = img - img_denoised
        noise_arr = rgb2gray(noise_arr)
                
        imsave(arr=noise_arr, fname= f'./noise/{mode}/' + path)
            
    except Exception as e:
        print(e)
      
