import os, shutil

from src.noise import extract_noise
from src.reference import extract_reference
from src.correlation import extract_correlation
from src.experiment import experiment


def main():
    models_cam = os.listdir('./dataset/')

    #extract_noise(models_cam) # ALL DATASET
    #extract_reference(models_cam) # TRAIN 
    #extract_correlation(models_cam) # TEST
    experiment(models_cam) # EXECUTE ALGORITHM
            
if __name__ == '__main__':
    main()