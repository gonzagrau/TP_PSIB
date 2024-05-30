import numpy as np
#import SimpleITK as sitk
import matplotlib.pyplot as plt
import cv2
import scipy
from scipy import signal
import scipy.ndimage as snd
#from skimage import data

path = r"C:\Users\ibajl\Desktop\TP PSIB\TP_PSIB\Dataset for Fetus Framework\Dataset for Fetus Framework\External Test Set\Standard\1372.png"

""" img = sitk.ReadImage(path)
ima = sitk.GetArrayFromImage(img)
# se pasa a un arreglo
plt.figure(figsize=(10,10))
plt.imshow(ima, cmap="gray",vmin=0, vmax=255)
plt.title('Imagen Original', fontsize=15)
plt.show()
 """

 # morfologia matematica 
#invierto la imagen 
imagen = cv2.imread(path, 0)
umbral, imagen_binaria = cv2.threshold(imagen, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) #otsu
imagen_binaria_inv = (imagen_binaria==0)
imagen_binaria_inv = imagen_binaria_inv.astype("uint8")



#eleccion del kernel 
kernel = np.ones((5, 5), 'uint8')

#dilatacion
dilate_img = cv2.dilate(imagen_binaria_inv, kernel, iterations=1)

""" plt.figure(figsize=(12,12))
plt.subplot(121),plt.imshow(imagen_binaria_inv,cmap='gray'),plt.title('Imagen binaria',fontsize=16)
plt.subplot(122),plt.imshow(dilate_img,cmap='gray'), plt.title('Imagen binaria dilatada',fontsize=16)
plt.show() """

#erocion 
erode_img = cv2.erode(imagen_binaria_inv, kernel, iterations=1)

""" plt.figure(figsize=(12,12))
plt.subplot(121),plt.imshow(imagen_binaria_inv,cmap='gray'),plt.title('Imagen binaria',fontsize=16)
plt.subplot(122),plt.imshow(erode_img,cmap='gray'), plt.title('Imagen binaria erosionada',fontsize=16)
plt.show()  """
 

#dilatacion erocion = close
close_img = cv2.erode(dilate_img, kernel, iterations=1)
plt.figure(figsize=(12,12))
plt.subplot(121),plt.imshow(imagen_binaria_inv,cmap='gray'),plt.title('Imagen binaria',fontsize=16)
plt.subplot(122),plt.imshow(erode_img,cmap='gray'), plt.title('Imagen binaria imagen con close',fontsize=16)
plt.show() 

""" s = np.linspace(0, 2*np.pi, 400) #Defino ángulos para armar el círculo inicial de 400 puntos
y = 150 + 30*np.cos(s) #Definimos puntos del círculo en el eje x
x = 175 + 30*np.sin(s) #Definimos puntos del círculo en el eje y
init = np.array([y, x]).T
# 150 y 175 son donde va a ir el circulo. 30 es el radio
# asi se pone el circulo para poder pasarlo, no se porque se transpone el array

plt.figure()
plt.plot(x, y)
plt.title('Circulo - Snake')
<<<<<<< HEAD
plt.show()
#ghhvhgfh
=======
plt.show()  """
>>>>>>> 6baf236cfc4900516696b03ba4f5fdbda6492198