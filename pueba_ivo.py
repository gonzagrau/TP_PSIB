import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dat_hea_reader import *
from eeg_avg import *
import scipy.signal as sig
import os

lista_paths=[]
directory = os.fsencode('base_de_datos_Prueba')

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".dat"):
        # Join the bytes path of the directory with the string filename
        filepath = os.path.join(directory, os.fsencode(filename))
        # Decode the filepath for printing
        name = os.fsdecode(filepath)
        lista_paths.append(name.strip('.dat'))

cant_files = len(lista_paths)
# se crea un indice para luego
i = 0

#para sacar datos sobre el tr_len y el tipo de dato
fs, trials, comments = read_trials(lista_paths[0])
tr_len = comments['Trial Length (samples)']

#se crean matrices de zeros que despues cada fila se rellena con el promeio de un archivo
mat_trials_mean = np.zeros((cant_files,tr_len),dtype=trials.dtype)
mat_trials_amp = np.zeros((cant_files,tr_len),dtype=trials.dtype)
mat_trials_var = np.zeros((cant_files,tr_len),dtype=trials.dtype)
mat_trials_both =np.zeros((cant_files,tr_len),dtype=trials.dtype)

#se recorren todos los archivos haciendo el priomedio de cada uno y llenando las matrices 
for file in lista_paths:
    try:
        fs, trials, comments = read_trials(file)

        tr_len = comments['Trial Length (samples)']
        t = np.linspace(0, tr_len/fs, tr_len)*1000 # convert to ms

        #para saver si los archivos tienen bien los datos

        """ trials_mean = average_EEG(trials, mode='homogenous') 
        print(f"la cantidad de nans en el file {file} es {np.sum(np.isnan(trials_mean))}") """
        
        
        mat_trials_mean[i,:] = mat_trials_mean

        trials_amp = average_EEG(trials, mode='amp')
        mat_trials_amp[i,:] = trials_amp

        trials_var = average_EEG(trials, mode='var')
        mat_trials_var[i,:] = trials_var

        trials_both = average_EEG(trials, mode='both')
        mat_trials_both[i,:] = trials_both

        #se suma 1 a el indice para ir a la siguiente fila
        i = i +1


    except ValueError:
        print(f"error con el archivo {file}")
        break




# Defino los headers de filas
row_headers = np.array([[100], [10], [15], [20], [25], [30], [35], [40], [45], [50], [55], [60], [65], [70], [75], [80], [85], [90], [95]])


# Concateno los headers 
data_con_header_homo = np.hstack((row_headers, mat_trials_mean))
data_con_header_amp = np.hstack((row_headers, mat_trials_amp))
data_con_header_var = np.hstack((row_headers, mat_trials_var))
data_con_header_both = np.hstack((row_headers, mat_trials_both))


for data in [data_con_header_amp, data_con_header_homo, data_con_header_var, data_con_header_both]:
    data = data[data[:, 0].argsort()]

# mando a CSV
#IMPORTANTE el separador esta en ; y esta con 2 decimales en float el tipo de dato
np.savetxt("promedios_homo.csv", data_con_header_homo, delimiter=";", fmt='%0.2f')
np.savetxt("promedios_amp.csv", data_con_header_amp, delimiter=";", fmt='%0.2f')
np.savetxt("promedios_var.csv", data_con_header_var, delimiter=";", fmt='%0.2f')
np.savetxt("promedios_both.csv", data_con_header_both, delimiter=";", fmt='%0.2f')





