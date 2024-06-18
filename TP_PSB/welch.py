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
directory = os.fsencode(r'TP_PSB\data_avg_N1')
for file in os.listdir(directory):
        filename = os.fsdecode(file)        
        filepath = os.path.join(directory, os.fsencode(filename))
        name = os.fsdecode(filepath)
        lista_paths.append(name)
        
#genero una funcion que dado un path a un csv me devuelve la suma del welch para cada header
def suma_welch(path,
               fs: int = 48000):
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

    return PSD_sum_mat

# calculo la suma del welch
PSD_sum_mat_amp= suma_welch(lista_paths[0])
PSD_sum_mat_both= suma_welch(lista_paths[1])
PSD_sum_mat_homo= suma_welch(lista_paths[2])
PSD_sum_mat_var= suma_welch(lista_paths[3])




           


