import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import cv2
import scipy
from scipy import signal
import scipy.ndimage as snd
#from skimage import data

path = r"C:\Users\ibajl\Desktop\TP PSIB\TP_PSIB\Dataset for Fetus Framework\Dataset for Fetus Framework\External Test Set\Standard\1372.png"

img = sitk.ReadImage(path)
ima = sitk.GetArrayFromImage(img)
# se pasa a un arreglo
plt.figure(figsize=(10,10))
plt.imshow(ima, cmap="gray",vmin=0, vmax=255)
plt.title('Imagen Original', fontsize=15)
plt.show()