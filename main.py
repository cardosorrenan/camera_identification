import os, shutil
from src.noise import extract_noise_dataset
from src.reference import extract_reference
from src.correlation import extract_correlation
from src.correlation_bin import extract_correlation_bin
from src.exp_histogram import extract_exp_histogram
from src.testing import testing_new

from src.threshold_scipy import extract_threshold as scipy
from src.threshold_paper import extract_threshold as paper
from src.threshold_experiment import extract_threshold as experiment

def main():

    models_cam = os.listdir('./dataset')
    
    #shutil.rmtree('./noise', ignore_errors=True)
    #extract_noise_dataset(models_cam)
    
    #shutil.rmtree('./reference', ignore_errors=True)
    #extract_reference(models_cam)    
    
    #shutil.rmtree('./correlation', ignore_errors=True)
    #extract_correlation(models_cam)
    
    #shutil.rmtree('./correlation_bin', ignore_errors=True)
    #extract_correlation_bin(models_cam)
    
    #shutil.rmtree('./histogram', ignore_errors=True)
    #extract_exp_histogram(models_cam)
    
    #shutil.rmtree('./model', ignore_errors=True)
    paper(models_cam)
    #experiment(models_cam)
    
    
    
    

            
if __name__ == '__main__':
    main()