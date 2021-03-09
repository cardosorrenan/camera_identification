import os, shutil
from src.extract_noises import extract_noises
from src.extract_reference import extract_reference
from src.estimate_corr import estimate_corr


def main():
    #shutil.rmtree('./noises', ignore_errors=True)
    #shutil.rmtree('./references', ignore_errors=True)
    
    models_cam = os.listdir('./dataset')
    
    #for model in models_cam: 
    #    extract_noises(model)
    #    extract_reference(model)
        
    estimate_corr(models_cam)

            
if __name__ == '__main__':
    main()