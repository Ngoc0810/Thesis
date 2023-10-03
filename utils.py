import cv2
import numpy as np
# return colored image, greyscale layer
import urllib

from consts import *

def read_url_image(url):
    try:
        req = urllib.request.urlopen(url)
        arr = np.asarray(bytearray(req.read()), dtype = np.byte)
        image = cv2.imdecode(arr, -1) # 'Load it as it is'
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (WIDTH, HEIGHT))
        grey = cv2.cvtColor(image,  cv2.COLOR_RGB2GRAY)
        return image, grey
    except:
        return None, None
    
def read_image(path):
    if path[:4] == 'http':
        return read_url_image(path)
    try:
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (WIDTH, HEIGHT))
        grey = cv2.cvtColor(image,  cv2.COLOR_RGB2GRAY)
        return image, grey
    except:
        return None, None

# png -> [argb] -> [rgb]
# png -> [rgb] -> lossless
# jpeg -> [rgb]
# jpg -> [rgb] -> lossy compression 
