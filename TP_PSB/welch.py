import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import welch
from dat_hea_reader import *
from eeg_avg import *
import scipy.signal as sig
import os
#fs de las realizaciones
fs = 48000
lista_paths=[]
directory = os.fsencode('data_avg_N1')
for file in os.listdir(directory):
        filename = os.fsdecode(file)        
        filepath = os.path.join(directory, os.fsencode(filename))
        name = os.fsdecode(filepath)
        lista_paths.append(name)
        
#genero una funcion que dado un path a un csv me devuelve la suma del welch para cada header
def suma_welch(path,
               fs: int = 48000,
               plot: bool = False):
    #cargo la data
    df = pd.read_csv(path)
    headers = df.columns.values.tolist()
    n = len(headers)

    #genero una matriz para almacenar la data

    PSD_sum_mat = np.zeros((n,2))

    for  i in range(n):
        #  Welch's 
        spl = df[headers[i]]
        nper = int(len(spl) // 5)
        ## se decide hacer 5 ventanas e manera empirica ya que se realizaron varios tests y es el que mas suavisa al periodograma sin tener errores por promediacion 
        ##el overlap es el clasico del 50% ya que se considera que es el que da mejores resultados
        f, Pxx_den = welch(spl, fs, noverlap=nper//2, nperseg=nper)


        suma_PSD = sum(Pxx_den)
        PSD_sum_mat[i,0] = int(headers[i])
        PSD_sum_mat[i,1] = suma_PSD

    #normalizo por el valor mas alto
    max_per_col = np.max(PSD_sum_mat,axis=0)
    max = max_per_col[1]
    for i in range(n):  
        PSD_sum_mat[i,1] = PSD_sum_mat[i,1]/max
    
    #Grafica
    if plot == True:
         graf_intensidad_vs_welch_suma(PSD_sum_mat)

    return PSD_sum_mat

def graf_intensidad_vs_welch_suma(mat: np.array, metodo: str):

    # extraigo las intencidades y las sumas
    intensity = mat[:, 0]
    suma = mat[:, 1]

    print(intensity)
    # Plot
    plt.figure(figsize=(8, 6))  # Optional: Adjust the figure size
    plt.plot(intensity, suma, marker='o', linestyle='-', color='b', label='sumas de welch vs Intensidad')

    # Add labels and title
    plt.xlabel('Intensidad')
    plt.ylabel('sumas de Welch')
    plt.title(f"Promediado {metodo}")

    # Add grid
    plt.grid(True)

    # Add legend
    plt.legend()

    # Show plot
    plt.show()

# calculo la suma del welch
PSD_sum_mat_amp= suma_welch(lista_paths[0])
PSD_sum_mat_both= suma_welch(lista_paths[1])
PSD_sum_mat_homo= suma_welch(lista_paths[2])
PSD_sum_mat_var= suma_welch(lista_paths[3])

#almaceno las matrices 
matrices_suma = {"amp":PSD_sum_mat_amp,"both":PSD_sum_mat_both,"homo":PSD_sum_mat_homo,"var":PSD_sum_mat_var}
for key, value in matrices_suma.items():
     graf_intensidad_vs_welch_suma(value, key)









           


