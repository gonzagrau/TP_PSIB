import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dat_hea_reader import *
from eeg_avg import *
import scipy.signal as sig
import os
import pandas as pd

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

#para sacar datos sobre el tr_len y el tipo de dato
fs, trials, comments = read_trials(lista_paths[0])
tr_len = comments['Trial Length (samples)']

#se crean matrices de zeros que despues cada fila se rellena con el promeio de un archivo
mat_trials_mean = np.zeros((cant_files,tr_len),dtype=trials.dtype)
mat_trials_amp = np.zeros((cant_files,tr_len),dtype=trials.dtype)
mat_trials_var = np.zeros((cant_files,tr_len),dtype=trials.dtype)
mat_trials_both = np.zeros((cant_files,tr_len),dtype=trials.dtype)

#se recorren todos los archivos haciendo el priomedio de cada uno y llenando las matrices 
for i, file in enumerate(lista_paths):
    try:
        fs, trials, comments = read_trials(file)

        trials_mean = average_EEG(trials, mode='homogenous') 
        mat_trials_mean[i,:] = trials_mean

        trials_amp = average_EEG(trials, mode='amp')
        mat_trials_amp[i,:] = trials_amp

        trials_var = average_EEG(trials, mode='var')
        mat_trials_var[i,:] = trials_var

        trials_both = average_EEG(trials, mode='both')
        mat_trials_both[i,:] = trials_both


    except ValueError:
        print(f"error con el archivo {file}")
        break


# Defino los headers de filas y los nombres
row_headers = np.array([[100], [10], [15], [20], [25], [30], [35], [40], [45], [50], [55], [60], [65], [70], [75], [80], [85], [90], [95]])
filenames = ['homo', 'amp', 'var', 'both']

# Mando a csv
for data, fname in zip([mat_trials_mean, mat_trials_amp, mat_trials_var, mat_trials_both], filenames):
    head_data = np.column_stack((row_headers, data))
    df = pd.DataFrame(head_data)
    df = df.sort_values(df.columns[0], ascending=False)
    df.to_csv(f"promedios_{fname}.csv", index=False, header=False)






