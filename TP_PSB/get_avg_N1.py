import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dat_hea_reader import *
from eeg_avg import *
import scipy.signal as sig
import os
import pandas as pd
import re


def main():
    lista_paths=[]
    directory = os.fsencode('data_raw_N1')
    lista_SPL = []
    spl_pattern = re.compile(r'(raw_)(\d*)(_F)')

    print('Buscando archivos')
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".dat"):
            # Join the bytes path of the directory with the string filename
            filepath = os.path.join(directory, os.fsencode(filename))
            # Decode the filepath for printing
            name = os.fsdecode(filepath)
            lista_paths.append(name[:-4]) # Remove extension
            pattern = re.compile(r'(raw_)(\d*)(_F)')
            spl = re.search(pattern, name).group(2)
            lista_SPL.append(int(spl))
    
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
    print('Promediando...')
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


    # Mando a csv
    filenames = ['homo', 'amp', 'var', 'both']
    output_dir = 'data_avg_N1'
    row_names = np.array(lista_SPL)
    
    print('Guardando a csv...')
    for data, fname in zip([mat_trials_mean, mat_trials_amp, mat_trials_var, mat_trials_both], filenames):
        sorted_indices = row_names.argsort()[::-1]
        sorted_data = data[sorted_indices]
        df = pd.DataFrame(sorted_data.T)
        df.to_csv(rf"{output_dir}/promedios_{fname}.csv", index=False, header=row_names[sorted_indices])

    print('Listorti, José María.')

if __name__ == '__main__':
    main()






