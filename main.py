import os, shutil
from src.noise import extract_noise
from src.reference import extract_reference
from src.correlation import extract_correlation
from src.threshold import extract_threshold


def main():
    
    models_cam = os.listdir('./dataset')
    
    #shutil.rmtree('./noise', ignore_errors=True)
    #extract_noise(models_cam)
    
    #shutil.rmtree('./reference', ignore_errors=True)
    #extract_reference(models_cam)    
    
    #shutil.rmtree('./correlation', ignore_errors=True)
    #extract_correlation(models_cam)
    
    shutil.rmtree('./histogram', ignore_errors=True)
    extract_threshold(models_cam)

            
if __name__ == '__main__':
    main()