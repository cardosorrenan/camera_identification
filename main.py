import urllib.request
from urllib.error import HTTPError
import os


def download_imgs():
  models_cam = os.listdir('./urls_cameras/')
  models_cam.remove('good_jpgs')
  for model in models_cam:
    path = f'./urls_cameras/{model}/urls_final'
    path_dataset = f'./dataset/{model}'
    if not os.path.exists(path_dataset):
      os.makedirs(path_dataset)
    with open(path) as file: 
      images_url = file.read().split('\n')      
    images_url = images_url[0:10]
    for i, url in enumerate(images_url):
      if not os.path.exists(f'{path_dataset}/sample_{i}.jpg'):
        try:
          urllib.request.urlretrieve(url, f'{path_dataset}/sample_{i}.jpg')
          print(f'OK - {model} sample_{i}')
        except HTTPError:
          print(f'ERROR - {model} sample_{i}')
      else:
        print(f'EXIST - {model} sample_{i}')
  return True


def main():
  download_imgs() # download 10 photos of each model


if __name__ == '__main__':
  main()


