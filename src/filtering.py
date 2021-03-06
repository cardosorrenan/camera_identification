# -*- coding: utf-8 -*-
def get_sigma(coeff, W_values, sigma_0):
    values = []

    for w in W_values:
        x = (coeff**2 - sigma_0**2)/w**2
        x_sum = x.sum()
        values.append(x_sum if x_sum > 0 else 0)
        
    sigma = min(values)
        
    return sigma


def wiener_filter(coeff, sigma, sigma_0):
    filtered_coeff = coeff*(sigma/(sigma - sigma_0**2))
    return filtered_coeff
                       
                        

def apply_filter(coeff):
    sigma_0 = 5
    W = [3, 5, 7, 9]
    sigma = get_sigma(coeff, W, sigma_0)
    coeff_filtered = wiener_filter(coeff, sigma, sigma_0)
    return coeff_filtered